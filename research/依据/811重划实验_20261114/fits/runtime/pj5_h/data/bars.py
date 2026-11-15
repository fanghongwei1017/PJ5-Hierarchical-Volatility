"""Load one ``(symbol, day)`` minute file into a normalised bar table.

The vendor feed is ``D:\\data\\A股分钟数据\\sh\\<YYYYMMDD>\\kline_<code>.SH.csv``: UTF-8 with
BOM, CRLF, ten columns (``万得代码,交易所代码,日期,时间,开盘价,最高价,最低价,收盘价,成交量,成交额``),
timestamps as ``HHMMSSmmm`` integers that are **left-endpoint** bar labels, prices in
yuan (unadjusted), volume in **shares**, amount in **yuan**
(``docs/data/data_contract.txt`` §2).

This module normalises that into the frozen bar schema of
``configs/contracts/interfaces.json``::

    symbol, timestamp, trading_date, session_id, bar_minutes,
    open, high, low, close, volume, amount, source_id, quality_flags

plus three helper columns used by :mod:`pj5_h.data.build`:
``label_min`` (minutes of the day), ``is_terminal`` (the 11:30 / 15:00 prints) and
``label_str`` (``"09:30"``).

Cleaning rules (each one counted in ``bars.attrs["diag"]``, none of them silent):

* rows whose timestamp is ``00:00:00.000`` are corrupt and dropped (§10.5 -- 68 rows in
  the whole sample);
* duplicated timestamp rows are detected and collapsed (first occurrence kept);
* rows with a non-finite or non-positive price are dropped -- such a bar cannot serve as
  a return endpoint;
* labels outside the frozen grid (and outside the two terminal prints) are dropped;
* an ``OHLC``-inconsistent bar (``high < low`` or a price outside ``[low, high]``) is
  *kept* but flagged, because its close is still a traded price.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .sessions import (
    STRUCTURAL_ABSENT_LABELS,
    TERMINAL_MINUTES,
    minutes_to_label,
    session_of_minutes,
)

__all__ = [
    "DATA_ROOT",
    "MINUTE_ROOT",
    "BAR_COLUMNS",
    "CORE_LABELS",
    "GRID_LABELS",
    "minute_file_path",
    "load_day_symbol",
    "empty_bars",
]

DATA_ROOT = r"D:\data"
MINUTE_ROOT = os.path.join(DATA_ROOT, "A股分钟数据")

_VENDOR_COLUMNS = ["万得代码", "交易所代码", "日期", "时间", "开盘价", "最高价",
                   "最低价", "收盘价", "成交量", "成交额"]
_RENAME = {
    "万得代码": "symbol",
    "日期": "trading_date",
    "时间": "raw_time",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "成交额": "amount",
}

#: The 240 left-endpoint labels of a complete day (am 120 + pm 120).
CORE_LABELS = tuple(list(range(9 * 60 + 30, 11 * 60 + 30)) + list(range(13 * 60, 15 * 60)))
GRID_LABELS = CORE_LABELS + tuple(TERMINAL_MINUTES)
_CORE_SET = set(CORE_LABELS)
_GRID_SET = set(GRID_LABELS)

BAR_COLUMNS = [
    "symbol", "timestamp", "trading_date", "session_id", "bar_minutes",
    "open", "high", "low", "close", "volume", "amount", "source_id", "quality_flags",
    "label_min", "label_str", "is_terminal",
]

_MARKET_BY_SUFFIX = {".SH": "sh", ".SZ": "sz", ".BJ": "bj"}


def minute_file_path(symbol: str, date: str, root: Optional[str] = None) -> str:
    """Absolute path of the per-stock minute file for one trading day."""
    sym = str(symbol).strip().upper()
    mk = _MARKET_BY_SUFFIX.get(sym[-3:])
    if mk is None:
        raise ValueError("cannot infer market from symbol %r" % (symbol,))
    day = str(date).replace("-", "")
    base = root if root is not None else MINUTE_ROOT
    return os.path.join(base, mk, day, "kline_%s.csv" % sym)


def empty_bars() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in BAR_COLUMNS})
    df["timestamp"] = pd.to_datetime(pd.Series(dtype="datetime64[ns]"))
    for c in ("session_id", "bar_minutes", "label_min"):
        df[c] = df[c].astype("int64")
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = df[c].astype("float64")
    df["is_terminal"] = df["is_terminal"].astype("bool")
    df.attrs["diag"] = {}
    return df


def load_day_symbol(symbol: str, date: str, root: Optional[str] = None) -> pd.DataFrame:
    """Read and normalise one ``(symbol, day)`` minute file.

    Returns a :class:`pandas.DataFrame` with :data:`BAR_COLUMNS`, sorted by ``label_min``.
    ``bars.attrs["diag"]`` holds every cleaning counter, e.g.::

        {"status": "ok", "n_rows_raw": 241, "n_dropped_ts_zero": 0,
         "n_dropped_duplicate": 0, "n_dropped_nonfinite": 0, "n_dropped_nonpositive": 0,
         "n_dropped_off_grid": 0, "n_kept": 241, "has_terminal_1130": False,
         "has_terminal_1500": True, "n_zero_volume": 3, "n_ohlc_inconsistent": 0}

    Raises ``FileNotFoundError`` when the file is absent -- the caller must decide what a
    missing file means (contract §6.1: it is *not* a reliable suspension test).
    """
    path = minute_file_path(symbol, date, root=root)
    day = str(date).replace("-", "")
    diag: Dict[str, object] = {"status": "ok", "path": path}
    if not os.path.exists(path):
        diag["status"] = "missing_file"
        out = empty_bars()
        out.attrs["diag"] = diag
        raise FileNotFoundError(path)

    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False,
                      usecols=_VENDOR_COLUMNS)
    diag["n_rows_raw"] = int(len(raw))
    if raw.empty:
        diag["status"] = "empty_file"
        out = empty_bars()
        out.attrs["diag"] = diag
        return out

    df = raw.rename(columns=_RENAME)
    # the vendor symbol is authoritative for the file, keep the caller's spelling
    df["symbol"] = str(symbol).strip().upper()
    df["trading_date"] = str(day)
    # everything is read as text first: a single garbage cell must not abort the day
    for c in ("raw_time", "open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad_ts = df["raw_time"].isna()
    diag["n_dropped_bad_timestamp"] = int(bad_ts.sum())
    df = df[~bad_ts]
    if df.empty:
        diag["status"] = "empty_file"
        out = empty_bars()
        out.attrs["diag"] = diag
        return out
    for c in ("volume", "amount"):
        diag["n_missing_" + c] = int(df[c].isna().sum())
        df[c] = df[c].fillna(0.0)

    # ---- timestamps -------------------------------------------------------
    t = df["raw_time"].to_numpy(dtype=np.int64)
    hh = t // 10_000_000
    mm = (t // 100_000) % 100
    ss = (t // 1_000) % 100
    ms = t % 1_000
    label_min = (hh * 60 + mm).astype(np.int64)

    keep = np.ones(len(df), dtype=bool)
    # (a) corrupt 00:00:00.000 rows
    zero_ts = (t == 0) | (label_min == 0)
    n_ts_zero = int(zero_ts.sum())
    keep &= ~zero_ts
    # (b) anything with sub-minute or out-of-grid stamps
    on_grid = np.isin(label_min, np.fromiter(_GRID_SET, dtype=np.int64))
    off_grid = keep & ((ss != 0) | (ms != 0) | ~on_grid)
    n_off_grid = int(off_grid.sum())
    keep &= ~off_grid
    diag["n_dropped_ts_zero"] = n_ts_zero
    diag["n_dropped_off_grid"] = n_off_grid

    # (c) prices: finite and strictly positive
    cols = ["open", "high", "low", "close"]
    px = df[cols].to_numpy(dtype=np.float64)
    finite = np.isfinite(px).all(axis=1)
    positive = (px > 0).all(axis=1)
    n_nonfinite = int((keep & ~finite).sum())
    n_nonpositive = int((keep & finite & ~positive).sum())
    keep &= finite & positive
    diag["n_dropped_nonfinite"] = n_nonfinite
    diag["n_dropped_nonpositive"] = n_nonpositive

    # (d) duplicates on the surviving labels: keep the first occurrence
    sub = df.loc[keep, ["raw_time"]].copy()
    sub["label_min"] = label_min[keep]
    dup = sub.duplicated(subset=["label_min"], keep="first").to_numpy()
    n_dup = int(dup.sum())
    diag["n_dropped_duplicate"] = n_dup

    keep_idx = np.flatnonzero(keep)
    keep_idx = keep_idx[~dup]
    keep = np.zeros(len(df), dtype=bool)
    keep[keep_idx] = True

    diag["n_kept"] = int(keep.sum())
    if not keep.any():
        diag["status"] = "all_rows_dropped"
        out = empty_bars()
        out.attrs["diag"] = diag
        return out

    out = pd.DataFrame({
        "symbol": df.loc[keep, "symbol"].to_numpy(),
        "trading_date": df.loc[keep, "trading_date"].to_numpy(),
        "open": df.loc[keep, "open"].to_numpy(dtype=np.float64),
        "high": df.loc[keep, "high"].to_numpy(dtype=np.float64),
        "low": df.loc[keep, "low"].to_numpy(dtype=np.float64),
        "close": df.loc[keep, "close"].to_numpy(dtype=np.float64),
        "volume": df.loc[keep, "volume"].to_numpy(dtype=np.float64),
        "amount": df.loc[keep, "amount"].to_numpy(dtype=np.float64),
        "label_min": label_min[keep].astype(np.int64),
    })
    out["label_str"] = [minutes_to_label(m) for m in out["label_min"].to_numpy()]
    out["is_terminal"] = out["label_min"].isin(TERMINAL_MINUTES).to_numpy()
    out["session_id"] = [session_of_minutes(m) for m in out["label_min"].to_numpy()]
    out["session_id"] = out["session_id"].astype("int64")
    out["bar_minutes"] = np.where(out["is_terminal"].to_numpy(), 0, 1).astype("int64")
    base = pd.Timestamp(day)
    out["timestamp"] = base + pd.to_timedelta(out["label_min"].to_numpy(), unit="m")
    base_root = root if root is not None else MINUTE_ROOT
    try:
        rel = os.path.relpath(path, base_root).replace("\\", "/")
    except ValueError:      # different drive (unit tests use a temporary root)
        rel = os.path.basename(path)
    out["source_id"] = rel

    # ---- per-bar quality flags -------------------------------------------
    n_zero_vol = int((out["amount"].to_numpy() == 0.0).sum())
    o, h, l, c = (out[k].to_numpy() for k in ("open", "high", "low", "close"))
    tol = 1e-9 * np.maximum(1.0, np.abs(c))
    ohlc_bad = (h < l) | (c > h + tol) | (c < l - tol) | (o > h + tol) | (o < l - tol)
    flags = np.array(["ok"] * len(out), dtype=object)
    flags[out["is_terminal"].to_numpy()] = "terminal_row"
    zv = out["amount"].to_numpy() == 0.0
    flags[zv & ~out["is_terminal"].to_numpy()] = _append_flag(
        flags[zv & ~out["is_terminal"].to_numpy()], "zero_volume")
    flags[ohlc_bad & ~out["is_terminal"].to_numpy()] = _append_flag(
        flags[ohlc_bad & ~out["is_terminal"].to_numpy()], "ohlc_inconsistent")
    out["quality_flags"] = flags

    labels = set(out["label_min"].tolist())
    diag["has_terminal_1130"] = bool(TERMINAL_MINUTES[0] in labels)
    diag["has_terminal_1500"] = bool(TERMINAL_MINUTES[1] in labels)
    diag["n_zero_volume"] = n_zero_vol
    diag["n_ohlc_inconsistent"] = int(ohlc_bad.sum())
    diag["n_missing_core"] = int(len(_CORE_SET - labels))
    diag["n_missing_structural_absent"] = int(
        len(set(STRUCTURAL_ABSENT_LABELS) - labels))

    out = out.sort_values("label_min", kind="mergesort").reset_index(drop=True)
    out = out[BAR_COLUMNS]
    out.attrs["diag"] = diag
    return out


def _append_flag(arr: np.ndarray, token: str) -> np.ndarray:
    return np.array([token if v == "ok" else "%s;%s" % (v, token) for v in arr], dtype=object)
