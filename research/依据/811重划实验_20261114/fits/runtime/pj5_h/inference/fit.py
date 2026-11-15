"""Windowing (Eq. 23) and the joint fit (Eq. 24) of PRE 109, 044313.

Two distinct objects are kept strictly apart, as required by the T00 contract:

* ``M``  the width of the rolling variance window used to *observe* a background series.
        It is an estimator choice, not a physical layer count.
* ``N``  the number of levels in the model hierarchy.  It is a model parameter.

The window is built **inside a session only**.  A rolling window that crosses the lunch
break or the overnight gap would mix two different trading clocks into one variance, which
is exactly the effect the extension is trying to measure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .. import status
from ..models import background_pdf, return_logpdf, return_pdf
from .binning import BinningSpec


# ---------------------------------------------------------------------------------------
# Eq. (23): rolling variance inside sessions
# ---------------------------------------------------------------------------------------
def rolling_variance(returns: np.ndarray, M: int, session_ids: np.ndarray | None = None):
    """v_M(t) = 1/(M-1) sum_{j=0}^{M-1} (x(t-j) - mean)^2, evaluated inside sessions only.

    Returns the array of window variances in the order the windows close, plus the index of
    each window's last observation.
    """
    x = np.asarray(returns, dtype=float)
    n = x.size
    if session_ids is None:
        session_ids = np.zeros(n, dtype=int)
    session_ids = np.asarray(session_ids)
    out = []
    idx = []
    if M < 2 or M > n:
        return np.empty(0), np.empty(0, dtype=int)
    # boundaries between sessions
    breaks = np.flatnonzero(np.diff(session_ids) != 0) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [n]])
    for s, e in zip(starts, ends):
        seg = x[s:e]
        if seg.size < M:
            continue
        # rolling mean/variance within the segment
        c1 = np.concatenate([[0.0], np.cumsum(seg)])
        c2 = np.concatenate([[0.0], np.cumsum(seg * seg)])
        k = np.arange(M, seg.size + 1)
        s1 = c1[k] - c1[k - M]
        s2 = c2[k] - c2[k - M]
        mean = s1 / M
        var = (s2 - M * mean * mean) / (M - 1)
        var = np.maximum(var, 0.0)
        out.append(var)
        idx.append(s + k - 1)
    if not out:
        return np.empty(0), np.empty(0, dtype=int)
    return np.concatenate(out), np.concatenate(idx)


def compound_gaussian_logpdf(x_grid: np.ndarray, v: np.ndarray) -> np.ndarray:
    """P^(M)(x) in Eq. (23): the Gaussian compounded with the empirical law of v."""
    x = np.atleast_1d(np.asarray(x_grid, dtype=float))
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return np.full(x.shape, -np.inf)
    vv = np.maximum(v, 1e-300)
    logk = (-0.5 * (x ** 2)[:, None] / vv[None, :]
            - 0.5 * np.log(2 * math.pi * vv)[None, :])
    mmax = logk.max(axis=1, keepdims=True)
    return mmax[:, 0] + np.log(np.exp(logk - mmax).mean(axis=1))


# ---------------------------------------------------------------------------------------
# Eq. (24): joint histogram log error
# ---------------------------------------------------------------------------------------
@dataclass
class FitError:
    total: float
    term_returns: float
    term_background: float
    n_return_bins_used: int
    n_v_bins_used: int
    n_return_bins_dropped: int
    n_v_bins_dropped: int
    status: str = status.OK


def eq24_error(x: np.ndarray, v: np.ndarray, N: int, beta: float, spec: BinningSpec,
               empty_bin_floor: float | None = None) -> FitError:
    """Error(N, beta) of Eq. (24).

    ``empty_bin_floor`` != None replaces the drop rule by a floor (sensitivity analysis).
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    xc = spec.x_centers
    vc = spec.v_centers
    cx, px = spec.histogram(spec.x_edges, x)
    lv = np.log(v[np.isfinite(v) & (v > 0)])
    cvd, _ = np.histogram(lv, bins=spec.v_edges_log)
    widths_v = spec.v_widths_log * vc          # d(ln v) -> dv at the centre
    pv = cvd / (cvd.sum() * widths_v) if cvd.sum() else np.zeros_like(vc)

    logp_model, stx = return_logpdf(xc, N, beta)
    logf_model, stv = background_pdf(vc, N, beta)
    logf_model = np.log(np.maximum(logf_model, 1e-300))

    def one_term(counts, dens, logmodel):
        use = counts > 0
        dropped = int((~use).sum())
        if empty_bin_floor is not None:
            dens = np.maximum(dens, empty_bin_floor)
            use = np.ones_like(use, dtype=bool)
        if not np.any(use):
            return 0.0, 0, dropped
        val = np.abs(logmodel[use] - np.log(np.maximum(dens[use], 1e-300))) * counts[use]
        return float(val.sum()), int(use.sum()), dropped

    t1, n1, d1 = one_term(cx, px, logp_model)
    t2, n2, d2 = one_term(cvd, pv, logf_model)
    denom = 2.0 * x.size - v.size
    denom = denom if denom > 0 else max(1.0, x.size)
    st = status.OK
    if any(s != status.OK for s in stx) or any(s != status.OK for s in stv):
        st = status.NO_CONVERGENCE
    return FitError(total=(t1 + t2) / denom, term_returns=t1 / denom,
                    term_background=t2 / denom, n_return_bins_used=n1,
                    n_v_bins_used=n2, n_return_bins_dropped=d1, n_v_bins_dropped=d2,
                    status=st)


# ---------------------------------------------------------------------------------------
# window selection (Eq. 23) and hierarchy fit (Eq. 24)
# ---------------------------------------------------------------------------------------
def select_window(returns, session_ids, spec_builder, M_grid=(3, 5, 8, 12, 20, 30),
                  metric: str = "hist_log_error"):
    """Choose M* by how well the compounded Gaussian matches the empirical returns.

    The empirical bins are rebuilt for each M from the same training data (the v_edges are
    irrelevant for this step; only the x bins are used).
    """
    x = np.asarray(returns, dtype=float)
    rows = []
    for M in M_grid:
        v, _ = rolling_variance(x, int(M), session_ids)
        if v.size < 100:
            rows.append({"M": int(M), "n_windows": int(v.size), "score": None,
                         "note": "too few windows"})
            continue
        spec = spec_builder(x, v)
        logpm = compound_gaussian_logpdf(spec.x_centers, v)
        cx, px = spec.histogram(spec.x_edges, x)
        use = cx > 0
        score = float(np.sum(np.abs(logpm[use] - np.log(np.maximum(px[use], 1e-300))) * cx[use])
                      / max(1, cx[use].sum()))
        # a second, bin-free metric as a robustness check
        xs = np.sort(x)
        grid = np.quantile(x, np.linspace(0.005, 0.995, 199))
        cdf_model = _compound_cdf(grid, v)
        cdf_emp = np.searchsorted(xs, grid) / xs.size
        ks = float(np.max(np.abs(cdf_model - cdf_emp)))
        rows.append({"M": int(M), "n_windows": int(v.size),
                     "score": score, "ks": ks})
    if metric == "ks":
        key = "ks"
    else:
        key = "score"
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return None, rows
    best = min(valid, key=lambda r: r[key])
    return int(best["M"]), rows


def _compound_cdf(x_grid, v, n_sub: int = 512):
    """CDF of the compounded Gaussian, by averaging the Gaussian CDF over v."""
    from scipy.stats import norm

    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size > n_sub:
        v = v[np.linspace(0, v.size - 1, n_sub).astype(int)]
    g = np.asarray(x_grid)[:, None] / np.sqrt(v)[None, :]
    return norm.cdf(g).mean(axis=1)


def fit_beta(x, v, N: int, spec: BinningSpec, beta_bounds=(0.05, 20.0),
             empty_bin_floor: float | None = None):
    """Minimise Eq. (24) over beta for a fixed N (one parameter, as in the paper)."""
    calls = []

    def obj(log_beta):
        beta = float(math.exp(log_beta))
        e = eq24_error(x, v, N, beta, spec, empty_bin_floor)
        calls.append((beta, e.total, e.status))
        return e.total

    res = minimize_scalar(obj, bounds=(math.log(beta_bounds[0]), math.log(beta_bounds[1])),
                          method="bounded", options={"xatol": 2e-3})
    beta_hat = float(math.exp(res.x))
    best = eq24_error(x, v, N, beta_hat, spec, empty_bin_floor)
    lo, hi = beta_bounds
    return {
        "N": int(N),
        "beta": beta_hat,
        "error": best.total,
        "term_returns": best.term_returns,
        "term_background": best.term_background,
        "status": best.status,
        "boundary_hit": bool(beta_hat <= lo * 1.02 or beta_hat >= hi * 0.98),
        "n_evaluations": len(calls),
        "n_return_bins_dropped": best.n_return_bins_dropped,
        "n_v_bins_dropped": best.n_v_bins_dropped,
    }


def fit_hierarchy(x, v, spec: BinningSpec, N_grid=(1, 2, 3, 4, 5),
                  beta_bounds=(0.05, 20.0), empty_bin_floor: float | None = None):
    """Fit beta for every N and select N by the smallest Eq. (24) error."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    table = [fit_beta(x, v, N, spec, beta_bounds, empty_bin_floor) for N in N_grid]
    ok = [r for r in table if r["status"] == status.OK]
    best = min(ok, key=lambda r: r["error"]) if ok else None
    return best, table
