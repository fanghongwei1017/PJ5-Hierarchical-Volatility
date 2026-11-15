"""Intraday-of-day (time-of-day) seasonality estimation and application.

Frozen interface (``configs/contracts/interfaces.json``)::

    fit_seasonality(train_returns, frequency) -> scale_by_tod (serializable dict) + diagnostics

Rules, verbatim from the contract:

* estimated on the **training block only**; the function refuses any row whose ``block``
  is not ``train`` (and refuses rows without a ``block`` column unless
  ``allow_unlabelled=True`` is passed explicitly for unit tests);
* the scale per time-of-day slot is a **robust** scale (MAD, converted to a Gaussian
  equivalent by the factor 1.4826), never a standard deviation;
* the scale is normalised to the overall training scale, so a slot with ``s == 1`` has the
  typical intraday magnitude and the numbers are dimension-free;
* the lower bound of ``s`` is decided **on the training block only** and frozen with the
  dict; nothing is re-estimated when the dict is applied to valid/test;
* the raw and the deseasonalised return are both kept, so ablation is possible.

The deseasonalised return is ``r_deseas = r_raw / s(tod)``.  Because ``s`` is a *scale*
and not a mean, applying it does not change the sign or the zero-return status of any
observation.

Why this matters for the H-theory fit (equation contract §5): a deterministic intraday
U-shape is itself a multi-scale structure and would be misread as an extra cascade layer,
so deseasonalisation must happen before any density fit, and it must not look at the
block that is being scored.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "MAD_TO_SIGMA",
    "SEASONALITY_SCHEMA",
    "fit_seasonality",
    "apply_seasonality",
    "scale_lookup",
    "seasonality_amplitude",
]

#: 1 / Phi^{-1}(0.75): converts a MAD into a Gaussian-equivalent standard deviation
MAD_TO_SIGMA = 1.4826

SEASONALITY_SCHEMA = "pj5.seasonality.v1"

#: a slot needs at least this many training returns before its own MAD is used
MIN_SLOT_N = 200
#: the floor is ``max(FLOOR_MIN, FLOOR_FRAC * median(s_raw))``, decided on train only
FLOOR_FRAC = 0.20
FLOOR_MIN = 0.05


def _robust_sigma(x: np.ndarray) -> Tuple[float, float]:
    """(robust sigma from the MAD, raw MAD).  Falls back to the mean absolute deviation."""
    if x.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = MAD_TO_SIGMA * mad
    if sigma <= 0.0:
        sigma = 1.2533 * float(np.mean(np.abs(x - med)))
    return sigma, mad


def fit_seasonality(train_returns, frequency: int, *,
                    allow_unlabelled: bool = False,
                    min_slot_n: Optional[int] = None) -> Tuple[Dict, Dict]:
    """Estimate ``s(tod)`` at one frequency from the training block only.

    Parameters
    ----------
    train_returns
        DataFrame with at least ``raw_log_return`` (or ``log_return``) and ``tod``; if a
        ``block`` column is present every row must be ``"train"``.
    frequency
        Bar length in minutes (only used for labelling / sanity checks).
    allow_unlabelled
        Only for unit tests: accept a frame without a ``block`` column.
    min_slot_n
        Minimum number of training observations for a slot to get its own MAD; defaults
        to :data:`MIN_SLOT_N` (200).  Lower it only for short smoke runs, and record it.

    Returns
    -------
    (scale_by_tod, diagnostics)
        ``scale_by_tod`` is a JSON-serialisable dict (schema :data:`SEASONALITY_SCHEMA`),
        ``diagnostics`` holds the amplitude, the floor decision and per-slot detail.
    """
    k = int(frequency)
    n_min = int(MIN_SLOT_N if min_slot_n is None else min_slot_n)
    if train_returns is None or len(train_returns) == 0:
        raise ValueError("fit_seasonality needs a non-empty training frame")
    df = train_returns
    col = "raw_log_return" if "raw_log_return" in df.columns else "log_return"
    if col not in df.columns:
        raise KeyError("train_returns must carry raw_log_return or log_return")
    if "block" in df.columns:
        blocks = set(df["block"].dropna().astype(str).unique())
        if blocks - {"train"}:
            raise ValueError(
                "fit_seasonality refuses non-training rows: found blocks %s "
                "(leakage guard, interfaces.json modules.data.fit_seasonality)"
                % (sorted(blocks),))
    elif not allow_unlabelled:
        raise ValueError(
            "train_returns has no 'block' column; pass allow_unlabelled=True only in tests")

    r = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    tod = df["tod"].astype(str).to_numpy() if "tod" in df.columns else None
    if tod is None:
        raise KeyError("train_returns must carry a 'tod' column (HH:MM of the bar start)")
    ok = np.isfinite(r)
    r, tod = r[ok], tod[ok]
    if r.size == 0:
        raise ValueError("no finite training returns")

    global_sigma, global_mad = _robust_sigma(r)
    if not np.isfinite(global_sigma) or global_sigma <= 0:
        raise ValueError("degenerate training dispersion (robust sigma == 0)")

    session_col = (df["session_id"].astype(int).to_numpy()[ok]
                   if "session_id" in df.columns else None)
    slot_idx = (df["slot_index"].astype(int).to_numpy()[ok]
                if "slot_index" in df.columns else None)

    slots: Dict[str, Dict] = {}
    order = []
    for label in pd.unique(pd.Series(tod)):
        m = tod == label
        x = r[m]
        sigma, mad = _robust_sigma(x)
        s_raw = float(sigma / global_sigma) if np.isfinite(sigma) else float("nan")
        entry = {
            "tod": str(label),
            "slot_index": int(slot_idx[m][0]) if slot_idx is not None else None,
            "session_id": int(session_col[m][0]) if session_col is not None else None,
            "n": int(x.size),
            "mad": float(mad),
            "sigma_robust": float(sigma),
            "mean_abs": float(np.mean(np.abs(x))),
            "median": float(np.median(x)),
            "s_raw": s_raw,
            "eligible": bool(x.size >= n_min and np.isfinite(s_raw) and s_raw > 0),
        }
        slots[str(label)] = entry
        order.append(str(label))
    order.sort(key=lambda t: (slots[t]["slot_index"] if slots[t]["slot_index"] is not None
                              else 10 ** 6, t))

    eligible = [slots[t]["s_raw"] for t in order if slots[t]["eligible"]]
    if not eligible:
        raise ValueError("no eligible time-of-day slot (all n < %d)" % n_min)
    floor = float(max(FLOOR_MIN, FLOOR_FRAC * float(np.median(eligible))))

    n_floor_hit = 0
    n_fallback = 0
    for t in order:
        e = slots[t]
        if not e["eligible"]:
            n_fallback += 1
            e["s"] = 1.0
            e["fallback"] = True
            e["floor_hit"] = False
            continue
        s = max(float(e["s_raw"]), floor)
        e["s"] = s
        e["fallback"] = False
        e["floor_hit"] = bool(s > e["s_raw"] + 1e-15)
        n_floor_hit += int(e["floor_hit"])

    s_all = np.array([slots[t]["s"] for t in order], dtype=np.float64)
    scale_by_tod = {
        "schema": SEASONALITY_SCHEMA,
        "frequency_minutes": k,
        "scale_kind": "mad_to_sigma_ratio",
        "mad_to_sigma": MAD_TO_SIGMA,
        "n_train_returns": int(r.size),
        "n_train_days": int(df["trading_date"].astype(str).nunique())
        if "trading_date" in df.columns else None,
        "train_day_first": str(df["trading_date"].astype(str).min())
        if "trading_date" in df.columns else None,
        "train_day_last": str(df["trading_date"].astype(str).max())
        if "trading_date" in df.columns else None,
        "global_sigma_robust": float(global_sigma),
        "global_mad": float(global_mad),
        "floor": floor,
        "floor_rule": "max(%.2f, %.2f * median(s_raw over eligible slots))"
                      % (FLOOR_MIN, FLOOR_FRAC),
        "min_slot_n": int(n_min),
        "slot_order": order,
        "slots": slots,
        "s_by_tod": {t: slots[t]["s"] for t in order},
    }
    diagnostics = {
        "frequency_minutes": k,
        "n_train_returns": int(r.size),
        "n_slots": len(order),
        "n_slots_eligible": int(len(eligible)),
        "n_slots_fallback": int(n_fallback),
        "n_slots_floor_hit": int(n_floor_hit),
        "floor": floor,
        "s_min": float(s_all.min()),
        "s_max": float(s_all.max()),
        "s_median": float(np.median(s_all)),
        "amplitude": float(s_all.max() / s_all.min()) if s_all.min() > 0 else float("inf"),
        "global_sigma_robust": float(global_sigma),
        "slots": [dict(slots[t]) for t in order],
    }
    return scale_by_tod, diagnostics


def scale_lookup(scale_by_tod: Dict) -> Dict[str, float]:
    """``{"09:30": 1.23, ...}`` from a fitted dict (works after a JSON round-trip)."""
    if "s_by_tod" in scale_by_tod:
        return {str(k): float(v) for k, v in scale_by_tod["s_by_tod"].items()}
    return {str(t): float(e["s"]) for t, e in scale_by_tod["slots"].items()}


def apply_seasonality(returns_df: pd.DataFrame, scale_by_tod: Dict, *,
                      default: float = 1.0) -> pd.DataFrame:
    """Return a copy with ``seasonality_scale`` and ``deseasonalized_log_return`` filled.

    The map is applied as-is (train-frozen); a time-of-day slot that was never seen in
    training gets ``default`` and the row is flagged in ``seasonality_fallback``.
    """
    out = returns_df.copy()
    mapping = scale_lookup(scale_by_tod)
    tod = out["tod"].astype(str)
    s = tod.map(mapping)
    missing = s.isna()
    if missing.any():
        s = s.where(~missing, default)
    out["seasonality_scale"] = s.astype(float).to_numpy()
    raw = pd.to_numeric(out["raw_log_return"], errors="coerce").to_numpy(dtype=np.float64)
    out["deseasonalized_log_return"] = raw / out["seasonality_scale"].to_numpy()
    out["seasonality_fallback"] = missing.to_numpy()
    return out


def seasonality_amplitude(scale_by_tod: Dict) -> float:
    """max(s) / min(s) over the fitted slots -- the size of the intraday effect."""
    s = np.array(list(scale_lookup(scale_by_tod).values()), dtype=np.float64)
    return float(s.max() / s.min()) if s.size and s.min() > 0 else float("nan")
