"""Aggregate 1-minute bars into k-minute bars and build the intraday return series.

Frozen interface (``configs/contracts/interfaces.json``)::

    build_returns(bars, session_rules, frequency) -> (returns_df, exclusions_df)

and the rules it must obey (``docs/data/data_contract.txt`` §4-§6,
``docs/theory/equation_contract.txt`` §5):

1. **Returns never cross a session boundary.**  A return is
   ``log(close_t / close_{t-1})`` between two *adjacent* k-minute bars of the same
   session; the first bar of a session produces no return.  The lunch break
   (11:30 -> 13:00), the overnight break and every interior missing bar are written to
   a separate gap table instead (see :func:`collect_gaps`), never to the main series.
   ``cross_session`` is therefore ``False`` on every row of the main series -- the column
   exists so that downstream code can assert it rather than trust it.
2. **80 % label-coverage rule.**  An interval ``[t, t+k)`` aggregates the minute bars
   that actually exist in it, but it is *dropped* -- and the drop is recorded in
   ``exclusions_df`` -- when fewer than 80 % of its required one-minute labels are
   present.  Required labels exclude the three structurally absent closing labels
   14:57/14:58/14:59 (:data:`pj5_h.data.sessions.STRUCTURAL_ABSENT_LABELS`); they are
   still aggregated when the vendor does provide them.  ``n_missing_labels`` counts all
   absent grid labels, ``n_missing_required`` only the required ones.
3. **The 15:00 terminal row belongs to the last afternoon bar.**  That bar covers
   ``[14:55, 15:00)`` at 5 minutes, is systematically missing 14:57-14:59, and its close
   must be the *official* close, which lives on the 15:00 row (contract §3.4a).  The row
   is merged into it and the bar gets ``includes_terminal_row``.  The 11:30 print is
   merged into the last morning bar **for volume only** (it is not a bar, §3.5, and
   unlike 15:00 it carries no official price) and flagged ``includes_am_terminal_row``.
4. **Zero returns are real observations.**  They stay in the series and are flagged:
   ``zero_return`` (close identical to the previous bar), ``zero_volume`` (the bar traded
   nothing, ``amount == 0``), ``suspended_zero_volume`` (the whole day traded nothing --
   contract §6.1 form A).
5. **Limit moves get sufficient-condition flags only.**  ``high == low == MAXUP`` or
   ``MAXDOWN`` from ``ASHAREAUCTION`` marks a locked bar (``limit_up_lock`` /
   ``limit_down_lock``); a zero-amplitude day without limit prices is marked
   ``limit_unknown``.  No censored likelihood, no deletion (contract §6.3).

Frequencies 1/5/15 (and 30/60) minutes are supported; every session is anchored on its
own start (09:30 for the morning, 13:00 for the afternoon), so a coarse bar never
straddles the lunch break.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .sessions import (
    interval_count,
    interval_label_minutes,
    interval_required_minutes,
    interval_starts,
    minutes_to_hhmmss,
    minutes_to_label,
    resolve_rules,
    session_bounds,
    session_ids,
    slot_index,
)

__all__ = [
    "RETURNS_COLUMNS",
    "RETURNS_BASE_COLUMNS",
    "EXCLUSION_COLUMNS",
    "GAP_COLUMNS",
    "MIN_LABEL_COVERAGE",
    "build_returns",
    "collect_gaps",
    "make_exclusion",
    "empty_returns",
    "empty_exclusions",
    "available_frequencies",
]

#: minimum share of required one-minute labels for a k-minute bar to survive
MIN_LABEL_COVERAGE = 0.8

#: columns frozen by ``interfaces.json`` (``modules.data.build_returns.returns_columns``)
RETURNS_BASE_COLUMNS = [
    "symbol", "interval_start", "interval_end", "trading_date", "session_id",
    "bar_minutes", "log_return", "cross_session", "missing_interior", "quality_flags",
]

#: everything :func:`build_returns` returns, frozen columns first
RETURNS_COLUMNS = RETURNS_BASE_COLUMNS + [
    "bar_start", "bar_end", "frequency_minutes", "session",
    "raw_log_return", "deseasonalized_log_return", "seasonality_scale",
    "zero_return", "zero_volume", "prev_zero_volume", "suspended_zero_volume",
    "includes_terminal_row", "includes_am_terminal_row", "no_terminal_row",
    "n_missing_labels", "n_missing_required", "n_required_labels", "n_bars_used",
    "tod", "slot_index", "open", "high", "low", "close", "prev_close", "volume", "amount",
    "block",
]

EXCLUSION_COLUMNS = [
    "symbol", "trading_date", "session_id", "frequency_minutes", "bar_start", "bar_end",
    "reason", "n_missing_labels", "n_required_labels", "n_bars_used", "detail",
]

GAP_COLUMNS = [
    "symbol", "frequency_minutes", "kind", "from_date", "from_time", "from_close",
    "to_date", "to_time", "to_open", "to_close", "log_change", "log_close_to_open",
    "n_missing_bars", "n_trading_days_ahead", "session_from", "session_to",
]

_SUPPORTED = (1, 5, 15, 30, 60)


def available_frequencies() -> Tuple[int, ...]:
    return _SUPPORTED


def empty_returns() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in RETURNS_COLUMNS})
    return df


def empty_exclusions() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in EXCLUSION_COLUMNS})


def make_exclusion(symbol, trading_date, session_id, frequency_minutes, bar_start,
                   bar_end, reason, n_missing_labels=0, n_required_labels=0,
                   n_bars_used=0, detail="") -> dict:
    """One row of ``exclusions_df``; ``None`` is allowed for day-level drops."""
    return {
        "symbol": symbol,
        "trading_date": str(trading_date),
        "session_id": session_id,
        "frequency_minutes": (int(frequency_minutes)
                              if frequency_minutes is not None else None),
        "bar_start": bar_start,
        "bar_end": bar_end,
        "reason": reason,
        "n_missing_labels": int(n_missing_labels),
        "n_required_labels": int(n_required_labels),
        "n_bars_used": int(n_bars_used),
        "detail": detail,
    }


def _normalise_limit_prices(limit_prices) -> Dict[str, Tuple[float, float]]:
    """Accept ``{date: (maxup, maxdown)}`` or a frame with ``trading_date/maxup/maxdown``."""
    if limit_prices is None:
        return {}
    if isinstance(limit_prices, dict):
        out = {}
        for k, v in limit_prices.items():
            try:
                up, dn = float(v[0]), float(v[1])
            except (TypeError, IndexError, ValueError):
                continue
            if math.isfinite(up) and math.isfinite(dn) and up > 0 and dn > 0:
                out[str(k).replace("-", "")] = (up, dn)
        return out
    if isinstance(limit_prices, pd.DataFrame):
        out = {}
        for r in limit_prices.itertuples(index=False):
            up, dn = float(getattr(r, "maxup")), float(getattr(r, "maxdown"))
            if math.isfinite(up) and math.isfinite(dn) and up > 0 and dn > 0:
                out[str(getattr(r, "trading_date")).replace("-", "")] = (up, dn)
        return out
    raise TypeError("limit_prices must be None, a dict or a DataFrame")


def build_returns(bars, session_rules=None, frequency: int = 5, *,
                  limit_prices=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate ``bars`` to ``frequency`` minutes and build within-session log returns.

    Parameters
    ----------
    bars
        Output of :func:`pj5_h.data.bars.load_day_symbol` -- one or more
        ``(symbol, trading_date)`` blocks; the frozen bar schema is required.
    session_rules
        ``"A_share_SH_SZ"`` (default) or the rules dict; validated, not reinterpreted.
    frequency
        Bar length in minutes, one of 1/5/15/30/60 (must divide the 120-minute session).
    limit_prices
        Optional ``{trading_date: (MAXUP, MAXDOWN)}`` for sufficient-condition limit
        flags.  Without it every zero-amplitude day is flagged ``limit_unknown``.

    Returns
    -------
    (returns_df, exclusions_df)
        ``returns_df`` has :data:`RETURNS_COLUMNS`; ``exclusions_df`` has
        :data:`EXCLUSION_COLUMNS` and lists *every* dropped interval with its reason.
    """
    rules = resolve_rules(session_rules)
    k = int(frequency)
    if k not in _SUPPORTED:
        raise ValueError("frequency must be one of %r, got %r" % (_SUPPORTED, frequency))
    lim = _normalise_limit_prices(limit_prices)

    if bars is None or len(bars) == 0:
        return empty_returns(), empty_exclusions()
    need = {"symbol", "trading_date", "label_min", "open", "high", "low", "close",
            "volume", "amount", "is_terminal", "session_id"}
    missing = need - set(bars.columns)
    if missing:
        raise KeyError("bars is missing columns: %s" % (sorted(missing),))

    ret_rows: List[dict] = []
    exc_rows: List[dict] = []
    gap_rows: List[dict] = []

    for symbol, sym_df in bars.groupby("symbol", sort=True):
        prev_pm = None            # (start_min, close) of the previous available day
        prev_day = None
        day_groups = [(str(d), sub) for d, sub in
                      sym_df.groupby(sym_df["trading_date"].astype(str), sort=True)]
        day_pos = {d: i for i, (d, _) in enumerate(day_groups)}
        for day, day_df in day_groups:
            r, e, gp, last_pm = _build_one_day(
                symbol=str(symbol), day=day, day_df=day_df, k=k,
                rules=rules, limit=lim.get(day),
                prev_pm=prev_pm, prev_day=prev_day,
                n_days_ahead=(day_pos[day] - day_pos[prev_day]) if prev_day else None,
            )
            ret_rows.extend(r)
            exc_rows.extend(e)
            gap_rows.extend(gp)
            if last_pm is not None:
                # carry the last *available* afternoon close forward: a suspended day
                # between two sessions must produce one overnight gap with
                # n_trading_days_ahead > 1, not silence (contract 8.4).
                prev_pm = last_pm
                prev_day = day

    rets = pd.DataFrame(ret_rows, columns=RETURNS_COLUMNS)
    excs = pd.DataFrame(exc_rows, columns=EXCLUSION_COLUMNS)
    rets.attrs["gaps"] = pd.DataFrame(gap_rows, columns=GAP_COLUMNS)
    if len(rets):
        rets = rets.sort_values(["symbol", "trading_date", "session_id", "bar_start"],
                                kind="mergesort").reset_index(drop=True)
    if len(excs):
        excs = excs.sort_values(["symbol", "trading_date", "session_id", "bar_start"],
                                kind="mergesort").reset_index(drop=True)
    return rets, excs


# --------------------------------------------------------------------------- one day
def _build_one_day(symbol, day, day_df, k, rules, limit, prev_pm, prev_day,
                   n_days_ahead):
    """Build the bars of one day, then its within-session returns and its gaps.

    Returns ``(return_rows, exclusion_rows, gap_rows, last_pm)`` where ``last_pm`` is
    ``(start_min, close)`` of the last afternoon bar of this day (``None`` when the
    afternoon session produced no bar), which the caller uses for the overnight gap.
    """
    exc: List[dict] = []
    g = day_df.sort_values("label_min")
    labels = g["label_min"].to_numpy()
    core = g[~g["is_terminal"].to_numpy()]
    term = g[g["is_terminal"].to_numpy()]

    row_by_label: Dict[int, Tuple[float, float, float, float, float, float]] = {}
    for o, h, l, c, v, a, lab in zip(core["open"], core["high"], core["low"], core["close"],
                                     core["volume"], core["amount"],
                                     core["label_min"].to_numpy()):
        row_by_label[int(lab)] = (float(o), float(h), float(l), float(c), float(v), float(a))

    terminal_by_session: Dict[int, Tuple[int, float, float, float, float, float, float]] = {}
    for lab, sid, o, h, l, c, v, a in zip(term["label_min"].to_numpy(),
                                          term["session_id"].to_numpy(),
                                          term["open"], term["high"], term["low"],
                                          term["close"], term["volume"], term["amount"]):
        terminal_by_session[int(sid)] = (int(lab), float(o), float(h), float(l),
                                         float(c), float(v), float(a))

    day_zero_amount = bool(len(g) and float(g["amount"].sum()) == 0.0)
    day_flat = bool(len(core) and
                    float(core["high"].max()) - float(core["low"].min()) <= 1e-9)
    maxup, maxdown = (limit if limit else (float("nan"), float("nan")))
    # A zero-amplitude day without MAXUP/MAXDOWN can only be marked "limit unknown".
    # A day that did not trade at all is a suspension, not a lock, and is flagged
    # suspended_zero_volume instead (contract 6.1 vs 6.3).
    limit_unknown_day = day_flat and not limit and not day_zero_amount

    # A file that carries no minute bar at all (suspension days are sometimes written with
    # only the 15:00 terminal row) is one day-level exclusion, not one per interval.
    if len(core) == 0:
        exc.append(make_exclusion(
            symbol, day, None, k, None, None, "empty_day_no_minute_bars", 0, 0, 0,
            "file carries 0 minute bars and %d terminal row(s)" % len(term)))
        return [], exc, [], None
    bars_by_session: Dict[int, List[dict]] = {}
    for sid in session_ids():
        built: List[dict] = []
        n_int = interval_count(k, sid)
        starts = interval_starts(k, sid)
        _, sess_end = session_bounds(sid)
        for j, start in enumerate(starts):
            end = start + k
            is_last = (j == n_int - 1)
            labs = interval_label_minutes(k, start)
            req = interval_required_minutes(k, start)
            present_all = [m for m in labs if m in row_by_label]
            present_req = [m for m in req if m in row_by_label]
            n_missing = k - len(present_all)
            n_missing_req = len(req) - len(present_req)
            tstart = pd.Timestamp(day) + pd.Timedelta(minutes=int(start))
            tend = pd.Timestamp(day) + pd.Timedelta(minutes=int(end))

            if not req:   # only 14:57/14:58/14:59 at k=1: optional interval
                if not present_all:
                    exc.append(make_exclusion(
                        symbol, day, sid, k, tstart, tend,
                        "structurally_absent_label", n_missing, 0, 0,
                        "label %s is dropped by the vendor after 2018 (contract 3.6)"
                        % minutes_to_label(start)))
                    continue
            else:
                coverage = len(present_req) / len(req)
                if coverage < MIN_LABEL_COVERAGE:
                    reason = "missing_label" if k == 1 else "label_coverage_below_80pct"
                    exc.append(make_exclusion(
                        symbol, day, sid, k, tstart, tend, reason,
                        n_missing, len(req), len(present_req),
                        "required present %d/%d (coverage %.3f < %.2f)"
                        % (len(present_req), len(req), coverage, MIN_LABEL_COVERAGE)))
                    continue

            members = [row_by_label[m] for m in present_all]
            # The terminal print may only be merged into the interval that actually ends
            # at the session close.  At k = 1 that is the *single* interval [14:59,15:00)
            # / [11:29,11:30); merging it into an earlier minute bar (e.g. 14:57 when the
            # vendor dropped 14:58/14:59) would silently turn a 1-minute interval into a
            # 3-minute one.  For k >= 5 the last interval already spans up to the close.
            merge_terminal = is_last and (k > 1 or start == sess_end - 1)
            terminal = terminal_by_session.get(sid) if merge_terminal else None
            if not members and terminal is None:
                exc.append(make_exclusion(
                    symbol, day, sid, k, tstart, tend, "no_bars_in_interval",
                    n_missing, len(req), 0, "no minute bar and no terminal row"))
                continue

            o_ = members[0][0] if members else terminal[1]
            c_ = members[-1][3] if members else terminal[4]
            hi_ = max([m[1] for m in members] + ([terminal[2]] if terminal else []))
            lo_ = min([m[2] for m in members] + ([terminal[3]] if terminal else []))
            vol_ = sum(m[4] for m in members) + (terminal[5] if terminal else 0.0)
            amt_ = sum(m[5] for m in members) + (terminal[6] if terminal else 0.0)
            inc_term = bool(terminal is not None and sid == 2)
            inc_am_term = bool(terminal is not None and sid == 1)
            if inc_term:
                c_ = terminal[4]          # official close lives on the 15:00 row
            no_term = bool(is_last and terminal is None and sid == 2)

            # sufficient-condition limit lock
            flat = (hi_ - lo_) <= 1e-9 * max(1.0, abs(c_))
            lock_up = flat and limit is not None and abs(c_ - maxup) <= 1e-6
            lock_dn = flat and limit is not None and abs(c_ - maxdown) <= 1e-6

            built.append(dict(
                session_id=sid, j=j, start=start, end=end,
                tstart=tstart, tend=tend,
                open=float(o_), high=float(hi_), low=float(lo_), close=float(c_),
                volume=float(vol_), amount=float(amt_),
                n_missing_labels=int(n_missing), n_missing_required=int(n_missing_req),
                n_required_labels=int(len(req)), n_bars_used=len(members),
                includes_terminal_row=inc_term, includes_am_terminal_row=inc_am_term,
                no_terminal_row=no_term,
                limit_up_lock=bool(lock_up), limit_down_lock=bool(lock_dn),
                limit_unknown=bool(limit_unknown_day),
            ))
        bars_by_session[sid] = built

    # -------- returns inside each session --------------------------------
    ret_rows: List[dict] = []
    gaps: List[dict] = []
    first_bar = {}
    last_bar = {}
    for sid in session_ids():
        built = bars_by_session[sid]
        if not built:
            continue
        first_bar[sid] = built[0]
        last_bar[sid] = built[-1]
        if built[0]["j"] > 0:
            gaps.append(_gap_row(symbol, k, "missing_prefix", None, None,
                                 built[0]["start"], built[0]["close"],
                                 n_missing_bars=built[0]["j"], sid=sid))
        for prev, cur in zip(built[:-1], built[1:]):
            if cur["j"] != prev["j"] + 1:
                gaps.append(_gap_row(symbol, k, "missing_bar", prev, cur, None, None,
                                     n_missing_bars=cur["j"] - prev["j"] - 1, sid=sid))
                continue
            ret_rows.append(_return_row(symbol, day, k, prev, cur, day_zero_amount))

    # -------- cross-session / overnight gaps -----------------------------
    if 1 in first_bar and 2 in first_bar:
        gaps.append(_gap_row(symbol, k, "session_break", last_bar[1], first_bar[2],
                             None, None, n_missing_bars=0, sid=2))
    if prev_pm is not None and 1 in first_bar:
        to_open = float(first_bar[1]["open"])
        gaps.append({
            "symbol": symbol, "frequency_minutes": k, "kind": "overnight",
            "from_date": str(prev_day), "from_time": minutes_to_hhmmss(prev_pm[0]),
            "from_close": float(prev_pm[1]),
            "to_date": str(day), "to_time": minutes_to_hhmmss(first_bar[1]["start"]),
            "to_open": to_open, "to_close": float(first_bar[1]["close"]),
            "log_change": _safe_log(float(first_bar[1]["close"]), float(prev_pm[1])),
            "log_close_to_open": _safe_log(to_open, float(prev_pm[1])),
            "n_missing_bars": 0,
            "n_trading_days_ahead": (int(n_days_ahead) if n_days_ahead else 1),
            "session_from": 2, "session_to": 1,
        })

    # gap rows created here carry no date; the caller fills it in
    for row in gaps:
        if row["from_date"] is None and row["from_time"] is not None:
            row["from_date"] = str(day)
        if row["to_date"] is None and row["to_time"] is not None:
            row["to_date"] = str(day)

    last_pm = (int(last_bar[2]["start"]), float(last_bar[2]["close"])) if 2 in last_bar else None
    return ret_rows, exc, gaps, last_pm


def _safe_log(a: float, b: float) -> float:
    if not (a > 0 and b > 0):
        return float("nan")
    return math.log(a / b)


def _return_row(symbol, day, k, prev, cur, day_zero_amount) -> dict:
    lr = _safe_log(cur["close"], prev["close"])
    zero_ret = bool(cur["close"] == prev["close"])
    zero_vol = bool(cur["amount"] == 0.0)
    flags: List[str] = []
    if cur["includes_terminal_row"]:
        flags.append("includes_terminal_row")
    if cur["includes_am_terminal_row"]:
        flags.append("includes_am_terminal_row")
    if cur["no_terminal_row"]:
        flags.append("no_terminal_row")
    if cur["n_missing_required"] > 0 or cur["n_missing_labels"] > 0:
        flags.append("partial_window")
    if zero_ret:
        flags.append("zero_return")
    if zero_vol:
        flags.append("zero_volume")
    if prev["amount"] == 0.0:
        flags.append("prev_zero_volume")
    if day_zero_amount:
        flags.append("suspended_zero_volume")
    if cur["limit_up_lock"]:
        flags.append("limit_up_lock")
    if cur["limit_down_lock"]:
        flags.append("limit_down_lock")
    if cur["limit_unknown"]:
        flags.append("limit_unknown")
    if not flags:
        flags.append("ok")
    return {
        "symbol": symbol,
        "interval_start": prev["tend"],
        "interval_end": cur["tend"],
        "trading_date": str(day),
        "session_id": cur["session_id"],
        "bar_minutes": k,
        "log_return": lr,
        "cross_session": False,
        "missing_interior": bool(cur["n_missing_required"] > 0),
        "quality_flags": ";".join(flags),
        "bar_start": cur["tstart"],
        "bar_end": cur["tend"],
        "frequency_minutes": k,
        "session": "am" if cur["session_id"] == 1 else "pm",
        "raw_log_return": lr,
        "deseasonalized_log_return": float("nan"),
        "seasonality_scale": float("nan"),
        "zero_return": zero_ret,
        "zero_volume": zero_vol,
        "prev_zero_volume": bool(prev["amount"] == 0.0),
        "suspended_zero_volume": bool(day_zero_amount),
        "includes_terminal_row": bool(cur["includes_terminal_row"]),
        "includes_am_terminal_row": bool(cur["includes_am_terminal_row"]),
        "no_terminal_row": bool(cur["no_terminal_row"]),
        "n_missing_labels": int(cur["n_missing_labels"]),
        "n_missing_required": int(cur["n_missing_required"]),
        "n_required_labels": int(cur["n_required_labels"]),
        "n_bars_used": int(cur["n_bars_used"]),
        "tod": minutes_to_label(cur["start"]),
        "slot_index": int(slot_index(k, cur["session_id"], cur["start"])),
        "open": float(cur["open"]),
        "high": float(cur["high"]),
        "low": float(cur["low"]),
        "close": float(cur["close"]),
        "prev_close": float(prev["close"]),
        "volume": float(cur["volume"]),
        "amount": float(cur["amount"]),
        "block": None,
    }


def _gap_row(symbol, k, kind, prev, cur, start_min, close, n_missing_bars=0, sid=None):
    """One row of the gap table.

    ``log_change`` is close-to-close across the break (the same definition as the main
    series, so a user can splice), ``log_close_to_open`` is the pure jump at the first
    print after the break.
    """
    if kind in ("missing_bar", "session_break"):
        to_open = float(cur["open"])
        return {
            "symbol": symbol, "frequency_minutes": k, "kind": kind,
            "from_date": None, "from_time": minutes_to_hhmmss(prev["start"]),
            "from_close": float(prev["close"]),
            "to_date": None, "to_time": minutes_to_hhmmss(cur["start"]),
            "to_open": to_open, "to_close": float(cur["close"]),
            "log_change": _safe_log(cur["close"], prev["close"]),
            "log_close_to_open": _safe_log(to_open, prev["close"]),
            "n_missing_bars": int(n_missing_bars), "n_trading_days_ahead": 0,
            "session_from": prev["session_id"], "session_to": cur["session_id"],
        }
    # missing_prefix: nothing precedes the first surviving bar of the session
    sess_start = session_bounds(sid)[0]
    return {
        "symbol": symbol, "frequency_minutes": k, "kind": kind,
        "from_date": None, "from_time": minutes_to_hhmmss(int(sess_start)),
        "from_close": float("nan"),
        "to_date": None, "to_time": minutes_to_hhmmss(int(start_min)),
        "to_open": float("nan"), "to_close": float(close),
        "log_change": float("nan"), "log_close_to_open": float("nan"),
        "n_missing_bars": int(n_missing_bars), "n_trading_days_ahead": 0,
        "session_from": sid, "session_to": sid,
    }


def collect_gaps(gaps: Iterable[dict]) -> pd.DataFrame:
    """Materialise the gap list into the frozen :data:`GAP_COLUMNS` table."""
    rows = list(gaps)
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in GAP_COLUMNS})
    return pd.DataFrame(rows, columns=GAP_COLUMNS)

