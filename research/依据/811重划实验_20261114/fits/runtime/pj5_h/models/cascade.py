"""Static (separation-of-timescales limit) hierarchical cascade.

The model is PRE 109, 044313 Eqs. (2) and (5): at each level i, conditional on the slower
level v_{i-1} = u,

    v_i | u ~ InvGamma(shape = beta + 1, scale = beta * u).

Because the conditional mean equals u, every level has the same mean v0.  In the limit of
infinite scale separation the joint law is the product of independent multipliers

    v_N = v_0 * prod_{i=1}^N Y_i,     Y_i iid, mean 1, Y ~ InvGamma(beta+1, beta).

This is the static sampler only.  It is NOT a substitute for the finite-r SDE of Eq. (2).

Implementation notes
--------------------
* The density of v_N is obtained two ways: exactly via the Meijer-G form Eq. (7), and
  independently by convolving the density of ln Y in Fourier space.  The FFT route is the
  fast one used by the fitting code; the Meijer route is the reference used by tests.
* Two traps were found while building this (see audit/evidence/T00/formula_audit.json):
  the convolution must be aligned so that index 0 is the origin, and it yields the density
  of ln v, so f(v) = g(ln v) / v.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import special

from .. import status

# ---------------------------------------------------------------------------------------
# conditional level law
# ---------------------------------------------------------------------------------------


def invgamma_pdf(v, shape: float, scale: float):
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v)
    good = v > 0
    vg = v[good]
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        logp = (shape * math.log(scale) - special.gammaln(shape)
                - (shape + 1.0) * np.log(vg) - scale / vg)
        out[good] = np.exp(logp)
    return out


def invgamma_logpdf(v, shape: float, scale: float):
    v = np.asarray(v, dtype=float)
    out = np.full(v.shape, -np.inf)
    good = v > 0
    if np.any(good):
        vg = v[good]
        out[good] = (shape * math.log(scale) - special.gammaln(shape)
                     - (shape + 1.0) * np.log(vg) - scale / vg)
    return out


def invgamma_rvs(shape: float, scale: float, size, rng: np.random.Generator):
    """Exact sampler: if G ~ Gamma(shape, 1) then scale / G ~ InvGamma(shape, scale)."""
    g = rng.gamma(shape, 1.0, size=size)
    with np.errstate(divide="ignore"):
        return scale / g


def static_cascade_sample(n: int, N: int, beta: float, v0: float = 1.0,
                          seed: int | None = None) -> np.ndarray:
    """Sample v_N from the static cascade.  Mean of the result is v0 for every beta > 0."""
    rng = np.random.default_rng(seed)
    v = np.full(n, float(v0))
    for _ in range(N):
        g = rng.gamma(beta + 1.0, 1.0, size=n)
        v = v * (beta / g)          # multiplier Y has mean 1 by construction
    return v


# ---------------------------------------------------------------------------------------
# fast engine: density of ln(v_N / v0) by Fourier convolution
# ---------------------------------------------------------------------------------------


_GRID = {"M": 1 << 18, "ymin": -45.0, "ymax": 45.0}
_MULT_CACHE: dict = {}
_BG_CACHE: dict = {}
_CACHE_MAX = 96


def set_grid(M: int = 1 << 18, ymin: float = -45.0, ymax: float = 45.0):
    """Change the tabulation grid.  Used by the fitting code to trade accuracy for speed.

    Any change clears the caches.  The accuracy cost of a coarser grid must be re-measured
    (see tools/t04_selftest.py) before it is used for a reported result.
    """
    _GRID.update({"M": int(M), "ymin": float(ymin), "ymax": float(ymax)})
    _MULT_CACHE.clear()
    _BG_CACHE.clear()


def _multiplier_grid(beta: float):
    """Density of ln Y on a uniform grid, Y ~ InvGamma(beta+1, beta) (mean 1)."""
    key = round(float(beta), 10)
    if key in _MULT_CACHE:
        return _MULT_CACHE[key]
    M, ymin, ymax = _GRID["M"], _GRID["ymin"], _GRID["ymax"]
    h = (ymax - ymin) / M
    y = ymin + h * np.arange(M)
    v = np.exp(y)
    g1 = invgamma_pdf(v, beta + 1.0, beta) * v          # density of ln Y
    mass = float(g1.sum() * h)
    if not np.isfinite(mass) or mass <= 0:
        return None, None, status.OVERFLOW
    g1 = g1 / mass
    if len(_MULT_CACHE) > _CACHE_MAX:
        _MULT_CACHE.clear()
    _MULT_CACHE[key] = (y, g1, mass)
    return y, g1, mass


def _log_background_grid(N: int, beta: float):
    """Density g of ln(v_N / v0) on a uniform grid y, plus a status flag."""
    if beta <= 0:
        return None, None, status.INVALID_PARAM
    key = (round(float(beta), 10), int(N))
    if key in _BG_CACHE:
        return _BG_CACHE[key]
    y, g1, mass = _multiplier_grid(beta)
    if y is None:
        return None, None, status.OVERFLOW
    h = y[1] - y[0]
    M = len(y)
    g = np.fft.irfft(np.fft.rfft(np.fft.ifftshift(g1)) ** N, n=M) * h ** (N - 1)
    g = np.fft.fftshift(g)
    g = np.maximum(g, 0.0)
    tot = float(g.sum() * h)
    if not np.isfinite(tot) or tot <= 0:
        return None, None, status.NO_CONVERGENCE
    g = g / tot
    st = status.OK if 0.99 < mass <= 1.0000001 else status.NO_CONVERGENCE
    if len(_BG_CACHE) > _CACHE_MAX:
        _BG_CACHE.clear()
    _BG_CACHE[key] = (y, g, st)
    return y, g, st


def _subsample(y, g, target: int = 4096):
    step = max(1, len(y) // target)
    return y[::step], g[::step]


def background_pdf(v, N: int, beta: float, v0: float = 1.0):
    """Eq. (7) evaluated from the fast engine.  Returns (density, status_array)."""
    y, g, st = _log_background_grid(int(N), float(beta))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    out = np.zeros(v.shape)
    if st != status.OK:
        return out, np.full(v.shape, st, dtype=object)
    good = v > 0
    with np.errstate(divide="ignore"):
        gi = np.interp(np.log(v[good] / v0), y, g)
    out[good] = gi / (v[good] / v0) / v0          # f(v) = g(ln(v/v0)) / v
    sts = np.where(good, status.OK, status.UNDERFLOW).astype(object)
    return out, sts


def background_logpdf(v, N: int, beta: float, v0: float = 1.0):
    y, g, st = _log_background_grid(int(N), float(beta))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    out = np.full(v.shape, -np.inf)
    if st != status.OK:
        return out, np.full(v.shape, st, dtype=object)
    good = v > 0
    with np.errstate(divide="ignore"):
        gi = np.interp(np.log(v[good] / v0), y, g)
        out[good] = np.log(np.maximum(gi, 1e-300)) - np.log(v[good] / v0) - math.log(v0)
    sts = np.where(good, status.OK, status.UNDERFLOW).astype(object)
    return out, sts


# ---------------------------------------------------------------------------------------
# return density
# ---------------------------------------------------------------------------------------


def return_logpdf(x, N: int, beta: float, v0: float = 1.0):
    """log P_N(x) from the Normal x InvGamma mixture.

    P_N(x) = int_0^inf Normal(x; 0, v) f_N(v) dv, evaluated in the log-variable:

        log P(x) = logsumexp_j [ -x^2/(2 v_j) - log sqrt(2 pi v_j) + log g_j + log h ]

    with v_j = v0 exp(y_j).  This is stable far into the tail, where a plain sum would
    underflow, and it is the form required for hold-out log scores.
    """
    y, g, st = _log_background_grid(int(N), float(beta))
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if st != status.OK:
        return np.full(x.shape, -np.inf), np.full(x.shape, st, dtype=object)
    ys, gs = _subsample(y, g)
    h = ys[1] - ys[0]
    v = v0 * np.exp(ys)
    logw = np.log(np.maximum(gs, 1e-300)) + math.log(h)
    x2 = (x ** 2)[:, None]
    logk = -0.5 * x2 / v[None, :] - 0.5 * np.log(2.0 * math.pi * v[None, :])
    lm = logk + logw[None, :]
    mmax = lm.max(axis=1, keepdims=True)
    out = (mmax[:, 0] + np.log(np.exp(lm - mmax).sum(axis=1)))
    return out, np.full(x.shape, status.OK, dtype=object)


def return_pdf(x, N: int, beta: float, v0: float = 1.0):
    lp, st = return_logpdf(x, N, beta, v0)
    with np.errstate(under="ignore"):
        return np.exp(lp), st


def return_cdf(x, N: int, beta: float, v0: float = 1.0, n_grid: int = 4001,
               x_max: float = 60.0):
    """CDF by numerical integration of the mixture on a half-line (for tests/perf checks)."""
    xs = np.linspace(0.0, x_max, n_grid)
    lp, _ = return_logpdf(xs, N, beta, v0)
    dens = np.exp(lp)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(xs))])
    out = np.interp(np.atleast_1d(np.asarray(x, dtype=float)), xs, cum)
    return np.minimum(out, 1.0)


# ---------------------------------------------------------------------------------------
# closed forms for N = 1 (Eq. 16) -- used as the primary numerical anchor
# ---------------------------------------------------------------------------------------


def return_pdf_n1_closed(x, beta: float, v0: float = 1.0):
    x = np.asarray(x, dtype=float)
    pref = math.gamma(beta + 1.5) / math.gamma(beta + 1) / math.sqrt(2 * math.pi * beta * v0)
    return pref * (1.0 + x * x / (2 * beta * v0)) ** (-(beta + 1.5))


def return_logpdf_n1_closed(x, beta: float, v0: float = 1.0):
    x = np.asarray(x, dtype=float)
    return (math.lgamma(beta + 1.5) - math.lgamma(beta + 1)
            - 0.5 * math.log(2 * math.pi * beta * v0)
            - (beta + 1.5) * np.log1p(x * x / (2 * beta * v0)))


def return_kurtosis_theory(N: int, beta: float) -> float:
    """3*(1+1/(beta-1))^N; infinite for beta <= 1 (fourth moment does not exist)."""
    if beta <= 1.0:
        return math.inf
    return 3.0 * (1.0 + 1.0 / (beta - 1.0)) ** N
