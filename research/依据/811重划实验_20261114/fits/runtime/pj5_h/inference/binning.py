"""Histogram binning frozen at training time.

The paper's Eq. (24) compares model and empirical *histograms* through a logarithmic
error.  It does not specify the bins, the treatment of empty bins, or whether the empirical
values are densities or probabilities.  These choices are frozen here, recorded in the
config, and their sensitivity is measured in ``tools/t04_selftest.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np


def freedman_diaconis_bins(x: np.ndarray, lo: float, hi: float) -> int:
    x = np.asarray(x, dtype=float)
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    n = x.size
    if iqr <= 0 or hi <= lo:
        return 30
    width = 2.0 * iqr * n ** (-1.0 / 3.0)
    nb = int(round((hi - lo) / width)) if width > 0 else 30
    return int(min(max(nb, 20), 80))


@dataclass
class BinningSpec:
    """Frozen bin edges for the return histogram and the background histogram."""

    x_edges: np.ndarray
    v_edges_log: np.ndarray
    n_x_bins: int
    n_v_bins: int
    quantile_range: tuple[float, float] = (0.001, 0.999)
    source: str = "train"

    @property
    def x_centers(self) -> np.ndarray:
        return 0.5 * (self.x_edges[1:] + self.x_edges[:-1])

    @property
    def x_widths(self) -> np.ndarray:
        return np.diff(self.x_edges)

    @property
    def v_centers(self) -> np.ndarray:
        return np.exp(0.5 * (self.v_edges_log[1:] + self.v_edges_log[:-1]))

    @property
    def v_widths_log(self) -> np.ndarray:
        return np.diff(self.v_edges_log)

    def histogram(self, x, values) -> tuple[np.ndarray, np.ndarray]:
        """Return (counts, densities) for ``values`` on the given edges."""
        counts, _ = np.histogram(values, bins=x)
        widths = np.diff(x)
        dens = counts / (counts.sum() * widths) if counts.sum() else np.zeros_like(widths, dtype=float)
        return counts, dens

    def to_dict(self) -> dict:
        return {
            "x_edges": self.x_edges.tolist(),
            "v_edges_log": self.v_edges_log.tolist(),
            "n_x_bins": self.n_x_bins,
            "n_v_bins": self.n_v_bins,
            "quantile_range": list(self.quantile_range),
            "source": self.source,
            "convention": {
                "x_bins": "equal width in x, over the central quantile range of the training set",
                "v_bins": "equal width in ln v, over the central quantile range of the training set",
                "empirical_values": "densities (count / (n * bin width)), not probabilities",
                "empty_bins": "dropped from the sum, count recorded; a sensitivity run adds a floor",
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BinningSpec":
        return cls(x_edges=np.array(d["x_edges"]), v_edges_log=np.array(d["v_edges_log"]),
                   n_x_bins=int(d["n_x_bins"]), n_v_bins=int(d["n_v_bins"]),
                   quantile_range=tuple(d.get("quantile_range", (0.001, 0.999))),
                   source=d.get("source", "train"))

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "BinningSpec":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def build_binning(returns, v_series, n_v_bins: int = 30,
                  quantile_range: tuple[float, float] = (0.001, 0.999),
                  n_x_bins: int | None = None) -> BinningSpec:
    """Build frozen bins from TRAINING data only."""
    x = np.asarray(returns, dtype=float)
    v = np.asarray(v_series, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    qlo, qhi = quantile_range
    lo, hi = np.quantile(x, [qlo, qhi])
    lim = max(abs(lo), abs(hi))
    if lim <= 0 or not np.isfinite(lim):
        raise ValueError("degenerate return distribution")
    nb = n_x_bins if n_x_bins is not None else freedman_diaconis_bins(x, -lim, lim)
    x_edges = np.linspace(-lim, lim, nb + 1)
    lv = np.log(v)
    vlo, vhi = np.quantile(lv, [0.005, 0.995])
    if not np.isfinite(vlo) or not np.isfinite(vhi) or vhi <= vlo:
        raise ValueError("degenerate background distribution")
    v_edges = np.linspace(vlo, vhi, n_v_bins + 1)
    return BinningSpec(x_edges=x_edges, v_edges_log=v_edges, n_x_bins=nb, n_v_bins=n_v_bins,
                       quantile_range=quantile_range)
