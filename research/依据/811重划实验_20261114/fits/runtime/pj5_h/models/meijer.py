"""Reference implementation of the H-theory densities through Meijer G functions.

These are the slow, high-precision reference values.  They are used by tests to check the
fast engine in :mod:`pj5_h.models.cascade`.  Nothing in the fitting pipeline calls them in a
hot loop.

Equations (PRE 109, 044313, p.3):

    f(v_N) = 1/(v0 * w * Gamma(beta+1)) * G^{0,N}_{N,0}( -(beta+1) ; - | v_N/(v0 w) )   (7)
    P_N(x) = 1/(sqrt(2 pi w v0) * Gamma(beta+1))
             * G^{1,N}_{N,1}( -(beta+1/2) ; 0 | x^2/(2 w v0) )                          (12)

with w = prod_j beta_j = beta**N and Gamma(beta+1) = prod_j Gamma(beta_j+1) = Gamma(beta+1)**N.
The product reading of the bold-vector Gamma is verified numerically in
audit/evidence/T00/formula_audit.json (check C).
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np

from .. import status

_DPS = 40


def _meijerg_with_retry(upper, lower, z, dps=_DPS):
    """Call mpmath.meijerg, retrying at higher precision / more terms.

    mpmath occasionally fails to converge for very small or very large z.  A failure is
    reported as a status code, never silently turned into 0.
    """
    # mpmath cannot converge for some (z, parameter) combinations -- notably small z with
    # N>=2 and non-integer beta.  One higher-precision retry is attempted, then the failure
    # is reported as a status code.  (Asking for very large maxterms was found to crash the
    # interpreter, so it is not attempted.)
    for attempt_dps in (dps, 2 * dps):
        try:
            with mp.workdps(attempt_dps):
                return mp.meijerg(upper, lower, z), status.OK
        except OverflowError:
            return None, status.OVERFLOW
        except Exception:  # ValueError / NoConvergence
            continue
    return None, status.NO_CONVERGENCE


def background_pdf(v, N: int, beta: float, v0: float = 1.0, dps: int = _DPS):
    """Eq. (7).  Returns (density, status_array)."""
    v = np.atleast_1d(np.asarray(v, dtype=float))
    out = np.zeros(v.shape)
    st = np.full(v.shape, status.OK, dtype=object)
    if beta <= 0:
        return out, np.full(v.shape, status.INVALID_PARAM, dtype=object)
    w = mp.mpf(beta) ** N
    with mp.workdps(dps):
        denom = mp.mpf(v0) * w * mp.gamma(mp.mpf(beta) + 1) ** N
    for i, vi in enumerate(v):
        if not np.isfinite(vi) or vi <= 0:
            st[i] = status.UNDERFLOW
            continue
        z = mp.mpf(float(vi)) / (mp.mpf(v0) * w)
        g, s = _meijerg_with_retry([[mp.mpf(-beta - 1)] * N, []], [[], []], z, dps=dps)
        if s != status.OK:
            st[i] = s
            continue
        val = float(g / denom)
        if not np.isfinite(val):
            st[i] = status.OVERFLOW if math.isinf(val) else status.UNDERFLOW
            continue
        out[i] = val
    return out, st


def return_pdf(x, N: int, beta: float, v0: float = 1.0, dps: int = _DPS):
    """Eq. (12).  Returns (density, status_array)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros(x.shape)
    st = np.full(x.shape, status.OK, dtype=object)
    if beta <= 0:
        return out, np.full(x.shape, status.INVALID_PARAM, dtype=object)
    w = mp.mpf(beta) ** N
    with mp.workdps(dps):
        denom = mp.sqrt(2 * mp.pi * w * mp.mpf(v0)) * mp.gamma(mp.mpf(beta) + 1) ** N
    for i, xi in enumerate(x):
        if not np.isfinite(xi):
            st[i] = status.INVALID_PARAM
            continue
        z = mp.mpf(float(xi)) ** 2 / (2 * w * mp.mpf(v0))
        g, s = _meijerg_with_retry([[mp.mpf(-beta) - mp.mpf(1) / 2] * N, []],
                                   [[mp.mpf(0)], []], z, dps=dps)
        if s != status.OK:
            st[i] = s
            continue
        val = float(g / denom)
        if not np.isfinite(val):
            st[i] = status.OVERFLOW if math.isinf(val) else status.UNDERFLOW
            continue
        out[i] = val
    return out, st


def nf0_asymptotic_partial_sum(x, N: int, beta: float, v0: float = 1.0, K: int = 40,
                              dps: int = _DPS):
    """Partial sum of the _NF_0 representation, Eq. (14).

    Provided only so that tests can demonstrate that this route is *not* usable: for N = 1
    the radius of convergence is |z| < 1, and for N >= 2 it is zero.  Never use this as a
    density algorithm.
    """
    a = beta + 1.5
    with mp.workdps(dps):
        w = mp.mpf(beta) ** N
        z = -(mp.mpf(abs(float(x))) ** 2) / (2 * w * mp.mpf(v0))
        pref = mp.gamma(mp.mpf(beta) + mp.mpf(1.5)) ** N / (
            mp.gamma(mp.mpf(beta) + 1) ** N * mp.sqrt(2 * mp.pi * w * mp.mpf(v0)))
        term = mp.mpf(1)
        s = mp.mpf(1)
        for k in range(1, K + 1):
            term = term * (mp.mpf(a) + k - 1) ** N / k * z
            s += term
        try:
            return float(pref * s)
        except (OverflowError, ValueError):
            return float("inf")
