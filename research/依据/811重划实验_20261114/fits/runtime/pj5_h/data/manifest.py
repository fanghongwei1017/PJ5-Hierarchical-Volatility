"""Bookkeeping for the minute pipeline: block assignment, sha256 manifests, summaries.

Nothing here estimates anything; it only counts what :mod:`pj5_h.data.build` produced, so
that every number a downstream task uses can be traced to a file and a hash.

Block assignment follows ``data_catalog/split_manifest.json`` literally: the 731 trading
days are indexed 0..730, ``train = [0,438)``, ``valid = [438,584)``, ``test = [584,731)``,
and the **last five trading days of each nominal block are purge days** which enter no
block's statistics.  Purge days are still written to the returns files, labelled
``purge``, so that they can be excluded by construction.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "sha256_file",
    "write_json",
    "open_gz_writer",
    "json_default",
    "load_split_manifest",
    "purge_days",
    "assign_blocks",
    "frequency_summary",
    "exclusion_counts",
    "gap_summary",
    "missing_label_distribution",
    "BLOCK_ORDER",
]

BLOCK_ORDER = ("train", "valid", "test", "purge")


# --------------------------------------------------------------------------- io
def sha256_file(path: str, blocksize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(blocksize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("not JSON serialisable: %r" % (type(o),))


def write_json(path: str, obj) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=json_default)
        fh.write("\n")


class _DeterministicGzip:
    """Text-mode writer that gzips with ``mtime=0`` so the file hash is reproducible.

    ``gzip.open`` stamps the current time into the header, which makes two runs of the same
    pipeline produce different sha256 values for identical content.  The manifest is a
    deliverable, so the header is pinned instead.
    """

    def __init__(self, path: str, compresslevel: int = 9):
        import gzip
        import io
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._raw = open(path, "wb")
        self._gz = gzip.GzipFile(fileobj=self._raw, mode="wb",
                                 compresslevel=compresslevel, mtime=0)
        self._text = io.TextIOWrapper(self._gz, encoding="utf-8", newline="")

    def __enter__(self):
        return self._text

    def __exit__(self, *exc):
        self._text.close()
        self._gz.close()
        self._raw.close()
        return False


def open_gz_writer(path: str):
    """Open a deterministic gzip text stream (see :class:`_DeterministicGzip`)."""
    return _DeterministicGzip(path)


# --------------------------------------------------------------------------- blocks
def load_split_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def purge_days(split: Dict) -> List[str]:
    return [str(d) for d in split["purge_days"]["all"]]


def assign_blocks(dates: Sequence[str], split: Dict) -> np.ndarray:
    """Map ``YYYYMMDD`` trading dates to ``train`` / ``valid`` / ``test`` / ``purge``.

    The mapping is done on the **position of the date in the full 731-day sequence**, and
    the caller must pass that full ordered sequence.  Dates outside the frozen window get
    ``"outside_split"`` rather than silently landing in a block.
    """
    days = [str(d) for d in dates]
    order = sorted(set(days))
    pos = {d: i for i, d in enumerate(order)}
    bounds = split["partition_rule"]
    if not bounds:  # pragma: no cover - the manifest always carries the rule
        raise ValueError("split manifest lacks partition_rule")
    n_train = split["blocks"]["train"]["n_days_nominal"]
    n_valid = split["blocks"]["valid"]["n_days_nominal"]
    n_test = split["blocks"]["test"]["n_days_nominal"]
    purge = set(purge_days(split))
    out = []
    for d in days:
        i = pos.get(d)
        if i is None or i >= n_train + n_valid + n_test:
            out.append("outside_split")
            continue
        if d in purge:
            out.append("purge")
        elif i < n_train:
            out.append("train")
        elif i < n_train + n_valid:
            out.append("valid")
        else:
            out.append("test")
    return np.array(out, dtype=object)


# --------------------------------------------------------------------------- summaries
def _pct(x, n) -> float:
    return float(x) / float(n) if n else float("nan")


def frequency_summary(returns_df: pd.DataFrame, frequency: int,
                      n_days_total: Optional[int] = None) -> Dict:
    """Valid sample size, window count and zero-return structure at one frequency."""
    k = int(frequency)
    df = returns_df
    n = len(df)
    if n:
        days = df["trading_date"].astype(str)
        per_day = df.groupby(["symbol", days.rename("d")]).size()   # one (symbol, day) window
        n_days = int(days.nunique())
    else:
        per_day = pd.Series(dtype=float)
        n_days = 0
    blocks: Dict[str, Dict] = {}
    for b in BLOCK_ORDER + ("outside_split",):
        m = df["block"].astype(str) == b if n else pd.Series(dtype=bool)
        sub = df[m] if n else df
        nn = len(sub)
        blocks[b] = {
            "n_returns": int(nn),
            "n_days": int(sub["trading_date"].astype(str).nunique()) if nn else 0,
            "n_symbols": int(sub["symbol"].nunique()) if nn else 0,
            "zero_return": int(sub["zero_return"].sum()) if nn else 0,
            "zero_return_share": _pct(sub["zero_return"].sum(), nn) if nn else float("nan"),
            "zero_volume": int(sub["zero_volume"].sum()) if nn else 0,
            "zero_volume_share": _pct(sub["zero_volume"].sum(), nn) if nn else float("nan"),
            "suspended_zero_volume": int(sub["suspended_zero_volume"].sum()) if nn else 0,
            "mean_raw_return": float(sub["raw_log_return"].mean()) if nn else float("nan"),
            "std_raw_return": float(sub["raw_log_return"].std(ddof=1)) if nn > 1 else float("nan"),
            "mean_abs_raw_return": float(sub["raw_log_return"].abs().mean()) if nn else float("nan"),
            "n_at_limit_up_lock": int(sub["quality_flags"].astype(str)
                                      .str.contains("limit_up_lock").sum()) if nn else 0,
            "n_at_limit_down_lock": int(sub["quality_flags"].astype(str)
                                        .str.contains("limit_down_lock").sum()) if nn else 0,
            "n_limit_unknown": int(sub["quality_flags"].astype(str)
                                   .str.contains("limit_unknown").sum()) if nn else 0,
            "n_includes_terminal_row": int(sub["includes_terminal_row"].sum()) if nn else 0,
            "n_partial_window": int(sub["quality_flags"].astype(str)
                                    .str.contains("partial_window").sum()) if nn else 0,
        }
    out = {
        "frequency_minutes": k,
        "n_returns": int(n),
        "n_days": n_days,
        "n_days_total_reference": int(n_days_total) if n_days_total else None,
        "n_symbols": int(df["symbol"].nunique()) if n else 0,
        "n_windows": int(len(per_day)) if n else 0,
        "returns_per_day": {
            "unit": "returns per (symbol, day) window",
            "min": int(per_day.min()) if len(per_day) else 0,
            "median": float(per_day.median()) if len(per_day) else float("nan"),
            "mean": float(per_day.mean()) if len(per_day) else float("nan"),
            "max": int(per_day.max()) if len(per_day) else 0,
            "nominal_full_day": {1: 238, 5: 46, 15: 14, 30: 6, 60: 2}.get(k),
            "n_windows_below_nominal": int((per_day < {1: 238, 5: 46, 15: 14, 30: 6, 60: 2}
                                            .get(k, 0)).sum()) if len(per_day) else 0,
        },
        "zero_return_share": _pct(df["zero_return"].sum(), n) if n else float("nan"),
        "blocks": blocks,
    }
    return out


def missing_label_distribution(returns_df: pd.DataFrame, frequency: int) -> Dict:
    """Distribution of ``n_missing_labels`` / ``n_missing_required`` at one frequency."""
    k = int(frequency)
    out = {"frequency_minutes": k}
    if not len(returns_df):
        return out
    for col in ("n_missing_labels", "n_missing_required"):
        vc = returns_df[col].astype(int).value_counts().sort_index()
        out[col] = {str(int(kk)): int(vv) for kk, vv in vc.items()}
    out["share_any_missing_required"] = float(
        (returns_df["n_missing_required"].astype(int) > 0).mean())
    out["share_any_missing_label"] = float(
        (returns_df["n_missing_labels"].astype(int) > 0).mean())
    return out


def exclusion_counts(exclusions_df: pd.DataFrame) -> List[Dict]:
    """``[{reason, n, share, n_symbols, n_days}]`` sorted by count."""
    if exclusions_df is None or not len(exclusions_df):
        return []
    tot = float(len(exclusions_df))
    rows = []
    for reason, sub in exclusions_df.groupby("reason", sort=False):
        rows.append({
            "reason": str(reason),
            "n": int(len(sub)),
            "share": len(sub) / tot,
            "n_symbols": int(sub["symbol"].nunique()),
            "n_days": int(sub["trading_date"].astype(str).nunique()),
            "n_required_labels_missing_total": int(
                sub["n_missing_labels"].astype(float).sum()) if "n_missing_labels" in sub else 0,
        })
    rows.sort(key=lambda r: -r["n"])
    return rows


def gap_summary(gaps_df: pd.DataFrame) -> Dict:
    """Counts and magnitudes of the intervals that were kept out of the return series."""
    if gaps_df is None or not len(gaps_df):
        return {"n_gaps": 0, "by_kind": {}}
    out = {"n_gaps": int(len(gaps_df)), "by_kind": {}}
    for kind, sub in gaps_df.groupby("kind", sort=False):
        lc = pd.to_numeric(sub["log_change"], errors="coerce").dropna()
        out["by_kind"][str(kind)] = {
            "n": int(len(sub)),
            "n_with_log_change": int(lc.size),
            "mean_abs_log_change": float(lc.abs().mean()) if lc.size else float("nan"),
            "median_abs_log_change": float(lc.abs().median()) if lc.size else float("nan"),
            "p99_abs_log_change": float(lc.abs().quantile(0.99)) if lc.size else float("nan"),
            "max_abs_log_change": float(lc.abs().max()) if lc.size else float("nan"),
            "share_abs_gt_1pct": float((lc.abs() > 0.01).mean()) if lc.size else float("nan"),
            "total_missing_bars": int(pd.to_numeric(sub["n_missing_bars"],
                                                    errors="coerce").fillna(0).sum()),
        }
    return out
