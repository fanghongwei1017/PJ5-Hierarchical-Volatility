"""Diagnostics for the SDE layer.

Everything here is either an *exact* theory CDF (obtained from the frozen T02 static
densities by numerical integration -- no re-derivation, no new free parameters), or a
sample statistic with an explicit uncertainty statement.

Two uncertainty statements are used, and they are NOT interchangeable:

``seed_band``
    From replicate seeds (W1..W3, W5).  ``mean +- 2*sd/sqrt(n_seeds)``.  This is the
    Monte-Carlo uncertainty of the *statistic*, not of a single sample.

``cdf_diff_permutation_null``
    For the trading-clock zero hypothesis (W4), where a seed-level band is too coarse.  The
    independent unit is the *path*: paths are iid while samples inside one path are not, so
    the per-path CDF vector ``q_p(u)`` carries the whole within-path dependence.  Permuting
    whole paths between the two groups gives an exact null (under "same law") that keeps the
    within-path autocorrelation intact.

Environment constraint (see ``audit/evidence/T06/ENVIRONMENT_numpy_blas.md``)
---------------------------------------------------------------------------
This machine's numpy 2.5.4 BLAS **hard-crashes the interpreter** (exit code -1066598273, no
traceback) for reductions of length >~4000: ``np.dot``, ``np.matmul``/``@``, ``np.linalg.*``
(``eigh`` included) and parts of ``scipy.linalg``.  matplotlib dies the same way.  So this
module uses **no** BLAS call at all: covariance-like products go through ``np.einsum``, and
the simultaneous null band goes through a permutation test instead of a matrix square root.
Allowed and used: elementwise ufuncs, ``np.sum``, ``np.cumsum``, ``np.interp``, ``np.fft``,
``np.sort``/``np.searchsorted``, ``np.quantile``, ``scipy.special`` ufuncs.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import special, stats

from .. import status
from ..models import background_pdf

# ---------------------------------------------------------------------------------------
# exact theory CDFs, built on top of the frozen T02 static model
# ---------------------------------------------------------------------------------------

_LOG_GRID_MIN = -45.0
_LOG_GRID_MAX = 45.0
_LOG_GRID_N = 20001

#: nodes used when tabulating the return CDF (the tabulation is cached per (N, beta, v0))
_RET_V_NODES = 5001
_RET_X_NODES = 2001
_RET_X_MAX = 2000.0


@lru_cache(maxsize=64)
def _log_background_density(N: int, beta: float, v0: float):
    """Density ``g`` of ``ln(v_N / v0)`` on a uniform grid, from the public T02 engine.

    ``background_pdf`` returns f(v); the density of the log variable is ``v * f(v)``.
    The mass captured on the grid is recorded, never silently swallowed.
    """
    y = np.linspace(_LOG_GRID_MIN, _LOG_GRID_MAX, _LOG_GRID_N)
    v = np.exp(y)
    f, st = background_pdf(v, int(N), float(beta), float(v0))
    g = f * v
    mass = float(np.trapezoid(g, y))
    if not np.isfinite(mass) or mass <= 0:
        raise FloatingPointError("log-grid background density has non-positive mass")
    g = g / mass
    return y, g, mass, st


def log_grid_mass(N: int, beta: float, v0: float = 1.0) -> float:
    """Mass captured by the log grid before renormalisation (should be 1 to ~1e-5)."""
    return _log_background_density(int(N), float(beta), float(v0))[2]


def theory_background_cdf(v, N: int, beta: float, v0: float = 1.0):
    """Exact CDF of v_N from Eq. (7): F(v) = int_{-inf}^{ln(v/v0)} g dy."""
    y, g, _mass, _st = _log_background_density(int(N), float(beta), float(v0))
    dy = np.diff(y)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * dy)])
    vv = np.atleast_1d(np.asarray(v, dtype=float))
    out = np.zeros(vv.shape)
    good = vv > 0
    with np.errstate(divide="ignore"):
        out[good] = np.interp(np.log(vv[good] / v0), y, cum)
    return out


@lru_cache(maxsize=64)
def _return_cdf_table(N: int, beta: float, v0: float):
    """Tabulate ``F(x) - 1/2`` for x >= 0 on a power-spaced grid, then interpolate.

    ``F(x) = E_v[Phi(x/sqrt(v))]``.  Because the mixture is symmetric, ``F(-x) = 1 - F(x)``,
    so only the one-sided excess has to be stored.  The power spacing ``x = Xmax u^2.5``
    puts most nodes where the density is, which keeps linear interpolation error ~1e-6.
    """
    y, g, _mass, _st = _log_background_density(int(N), float(beta), float(v0))
    step = max(1, y.size // _RET_V_NODES)
    ys, gs = y[::step], g[::step]
    h = float(ys[1] - ys[0])
    w = gs * h
    s = np.sqrt(v0 * np.exp(ys))

    u = np.linspace(0.0, 1.0, _RET_X_NODES)
    xs = _RET_X_MAX * u ** 2.5
    exc = np.empty(xs.size)
    for i in range(0, xs.size, 128):
        chunk = xs[i:i + 128]
        arg = chunk[:, None] / s[None, :]
        exc[i:i + 128] = (special.ndtr(arg) * w[None, :]).sum(axis=1) - 0.5
    exc = np.maximum.accumulate(np.clip(exc, 0.0, 0.5))
    return xs, exc


def theory_return_cdf(x, N: int, beta: float, v0: float = 1.0):
    """Exact CDF of the static mixture, by table interpolation (safe on 1e6-point samples)."""
    xs, exc = _return_cdf_table(int(N), float(beta), float(v0))
    xx = np.asarray(x, dtype=float)
    a = np.abs(xx)
    e = np.interp(a, xs, exc)
    return np.where(xx >= 0.0, 0.5 + e, 0.5 - e)


def theory_return_pdf(x, N: int, beta: float, v0: float = 1.0):
    """Static mixture pdf, evaluated through the frozen T02 fast engine (declared units)."""
    from ..models import return_pdf
    return return_pdf(x, int(N), float(beta), float(v0))


# ---------------------------------------------------------------------------------------
# sample statistics
# ---------------------------------------------------------------------------------------


def cdf_max_distance(sample, theory_cdf):
    """One-sample Kolmogorov-Smirnov distance against a *known* (not estimated) CDF.

    Returns ``(distance, location, theory_cdf_at_location)``.
    """
    s = np.asarray(sample, dtype=float).ravel()
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    s = np.sort(s)
    F = np.asarray(theory_cdf(s), dtype=float)
    i = np.arange(1, n + 1, dtype=float)
    d = np.maximum(np.abs(i / n - F), np.abs((i - 1.0) / n - F))
    k = int(np.argmax(d))
    return float(d[k]), float(s[k]), float(F[k])


def ks_critical_scale(n: int) -> float:
    """0.87/sqrt(n): the *typical* KS distance of a correctly specified sample.

    Reported next to every distance so that a failure at 0.02 is judged against the intrinsic
    finite-sample floor rather than against zero.
    """
    return 0.87 / math.sqrt(max(1, n))


def ecdf_on_grid(sample, grid):
    grid = np.asarray(grid, dtype=float)
    s = np.sort(np.asarray(sample, dtype=float).ravel())
    return np.searchsorted(s, grid, side="right") / max(1, s.size)


def quantile_grid(sample, n_points: int = 201):
    """Evaluation grid on the sample's own quantiles (dense where the data are)."""
    s = np.asarray(sample, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.zeros(1)
    qs = np.linspace(0.0, 1.0, n_points)
    return np.quantile(s, qs)


def per_path_cdf(samples2d, grid):
    """Per-path CDF values ``q_p(u)`` on an evaluation grid.

    ``samples2d`` has shape (n_paths, n_samples); the result is (n_paths, n_grid).  Rows are
    iid across paths, which is exactly the independence structure the permutation null needs.
    """
    a = np.atleast_2d(np.asarray(samples2d, dtype=float))
    grid = np.asarray(grid, dtype=float)
    out = np.empty((a.shape[0], grid.size))
    for p in range(a.shape[0]):
        out[p] = ecdf_on_grid(a[p], grid)
    return out


def cdf_diff_permutation_null(qa, qb, level: float = 0.95, n_perm: int = 200,
                              seed: int = 12345):
    """Simultaneous ``level`` band for ``max_u |F_a(u) - F_b(u)|`` under "same law".

    Whole paths are permuted between the two groups, so the within-path autocorrelation of
    every statistic is preserved and no distributional assumption is made beyond path
    independence.  Replaces an earlier eigenvalue/Cholesky construction because
    ``np.linalg.eigh`` hard-crashes on this machine.
    """
    qa = np.atleast_2d(qa)
    qb = np.atleast_2d(qb)
    Pa = qa.shape[0]
    pooled = np.vstack([qa, qb])
    P = pooled.shape[0]
    if Pa < 2 or P - Pa < 2:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    stat = np.empty(int(n_perm))
    for b in range(int(n_perm)):
        idx = rng.permutation(P)
        ma = pooled[idx[:Pa]].mean(axis=0)
        mb = pooled[idx[Pa:]].mean(axis=0)
        stat[b] = float(np.max(np.abs(ma - mb)))
    return float(np.quantile(stat, level))


def two_sample_cdf_distance(a, b, grid=None, n_points: int = 401):
    """max |F_a - F_b| on a data-driven grid (pooled quantiles)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if grid is None:
        grid = np.quantile(np.concatenate([a, b]), np.linspace(0.0, 1.0, n_points))
    return float(np.max(np.abs(ecdf_on_grid(a, grid) - ecdf_on_grid(b, grid))))


# ---------------------------------------------------------------------------------------
# autocorrelation / effective sample size
# ---------------------------------------------------------------------------------------


def averaged_autocorrelation(series, max_lag_frac: float = 0.5):
    """Path-averaged autocorrelation of a (n_paths, n_time) array, by FFT.

    Averaging over independent paths is what makes the tail usable: one path has O(1) noise
    on rho_k, an average over P paths has O(1/sqrt P).
    """
    s = np.atleast_2d(np.asarray(series, dtype=float))
    P, T = s.shape
    if T < 4:
        return np.array([1.0])
    s = s - s.mean(axis=1, keepdims=True)
    n = int(2 ** math.ceil(math.log2(2 * T)))
    f = np.fft.rfft(s, n=n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), n=n, axis=1)[:, :T].real
    acov = acov.mean(axis=0)
    v0 = acov[0]
    if not np.isfinite(v0) or v0 <= 0:
        return np.array([1.0])
    return (acov / v0)[:max(1, int(max_lag_frac * T))]


def integrated_autocorr_time(series, c: float = 5.0, max_lag_frac: float = 0.5):
    """Integrated autocorrelation time by Sokal's automatic windowing.

    ``tau_int = 1/2 + sum_{k=1}^{K} rho_k`` with the smallest K such that ``K >= c*tau``.
    Returns ``(tau_int, K, rho)``; ``n_eff = N_total / (2 tau_int)`` in the same units as the
    sample spacing.
    """
    rho = averaged_autocorrelation(series, max_lag_frac=max_lag_frac)
    tau = 0.5
    K = 0
    for k in range(1, rho.size):
        tau += float(rho[k])
        K = k
        if k >= c * tau:
            break
    return float(tau), int(K), rho


def seed_band(values):
    """(mean, sd over seeds, 95% half-width of the mean)."""
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(a.mean())
    if a.size < 2:
        return m, float("nan"), float("nan")
    sd = float(a.std(ddof=1))
    return m, sd, float(2.0 * sd / math.sqrt(a.size))


def kurtosis_raw(values):
    """Raw kurtosis (3 for a Gaussian).  None when the fourth moment is not estimable."""
    a = np.asarray(values, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size < 10:
        return None
    m = float(a.mean())
    v = float(((a - m) ** 2).mean())
    if v <= 0:
        return None
    return float(((a - m) ** 4).mean() / (v * v))


# ---------------------------------------------------------------------------------------
# verdict helper -- the W2 applicability criterion, written down BEFORE the scan
# ---------------------------------------------------------------------------------------

FAIL_THRESHOLD = 0.02


def verdict_flag(distance_mean, band, threshold: float = FAIL_THRESHOLD) -> str:
    """Three-way verdict, never a two-way "significant / not".

    ``fails``         distance above threshold even after removing the MC band
    ``applies``       distance below threshold even after adding the MC band
    ``indeterminate`` the MC band straddles the threshold; reported as such, not resolved
    """
    if not (np.isfinite(distance_mean) and np.isfinite(band)):
        return "indeterminate"
    if distance_mean - band > threshold:
        return "fails"
    if distance_mean + band < threshold:
        return "applies"
    return "indeterminate"


def status_summary(status_array) -> dict:
    a = np.asarray(status_array, dtype=object).ravel()
    out: dict = {}
    for s in a:
        out[str(s)] = out.get(str(s), 0) + 1
    return out


__all__ = [
    "FAIL_THRESHOLD",
    "averaged_autocorrelation",
    "cdf_diff_permutation_null",
    "cdf_max_distance",
    "ecdf_on_grid",
    "integrated_autocorr_time",
    "ks_critical_scale",
    "kurtosis_raw",
    "log_grid_mass",
    "per_path_cdf",
    "quantile_grid",
    "seed_band",
    "status_summary",
    "theory_background_cdf",
    "theory_return_cdf",
    "theory_return_pdf",
    "two_sample_cdf_distance",
    "verdict_flag",
    "status",
    "stats",
]
