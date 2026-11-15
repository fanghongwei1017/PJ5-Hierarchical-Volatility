"""Deterministic dt-convergence diagnostic by common random numbers (CRM).

Why this module exists
----------------------
A Kolmogorov-Smirnov distance measured on one seed is a random variable with a floor of order
``0.87/sqrt(n_eff)``.  Near that floor a *decrease* with dt cannot be demonstrated from a single
run: the seed-to-seed scatter is the same size as the dt effect.  Comparing "is KS monotone in
dt" is therefore the wrong test and it fails intermittently for reasons that have nothing to do
with the integrator.

The fix used here is to remove the sampling noise instead of loosening the criterion.  Under
**common random numbers** the coarse scheme is driven by the *same Brownian path* as the fine
scheme:

    dW_coarse[j] = sqrt(dt) * (Z[2j] + Z[2j+1]),      dW_fine[2j] = sqrt(dt) Z[2j], ...

so the sum of the fine increments over a coarse step equals the coarse increment and both
discretisations see the same driving noise at the coarse time grid.  The remaining difference
between the two solutions is pure discretisation bias, with no Monte-Carlo noise beyond the
usual (tiny) averaging over paths.  The measured order is the strong order of the log-Euler
scheme for this SDE, and the slope lets us *prove* convergence rather than infer it from a
difference that is buried in noise.

Reported: ``results/boundary/crm_convergence.csv`` and the ``convergence`` section of
``results/boundary/numeric_failures.json``.
"""

from __future__ import annotations

import math

import numpy as np


def log_euler_path(v0: float, gammas, kappas, dt: float, dW: np.ndarray) -> np.ndarray:
    """Integrate the hierarchy in ``y = ln v`` for one fixed set of increments.

    ``dW`` has shape ``(n_steps, n_paths, N)`` and already includes ``sqrt(dt)``.
    Returns ``v`` with shape ``(n_steps + 1, n_paths, N)`` (index 0 is the initial state).
    """
    n_steps, n_paths, N = dW.shape
    g = np.asarray(gammas, dtype=float)
    k = np.asarray(kappas, dtype=float)
    c1 = g * dt
    c2 = 0.5 * k * k * dt
    log_v0 = math.log(v0)
    z = np.zeros((n_paths, N))
    out = np.empty((n_steps + 1, n_paths, N))
    out[0] = np.exp(z)
    dfrac = np.empty((n_paths, N))
    for t in range(n_steps):
        dfrac[:, 0] = log_v0 - z[:, 0]
        if N > 1:
            np.subtract(z[:, :-1], z[:, 1:], out=dfrac[:, 1:])
        ed = np.exp(dfrac)
        z = z + (c1 * ed - (c1 + c2)) + k * dW[t]
        out[t + 1] = np.exp(z)
    return out


def crm_pair(v0: float, gammas, kappas, dt: float, n_steps_fine: int, n_paths: int,
             seed: int, n_levels_noise: int | None = None):
    """One fine and one coarse solution driven by the same Brownian path (ratio 2)."""
    N = len(gammas)
    if n_levels_noise is None:
        n_levels_noise = N
    rng = np.random.default_rng(int(seed))
    n_fine = int(n_steps_fine)
    if n_fine % 2 != 0:
        n_fine += 1
    Z = rng.standard_normal((n_fine, int(n_paths), N))
    sq = math.sqrt(dt)
    dW_fine = Z * sq
    dW_coarse = (Z[0::2] + Z[1::2]) * sq
    v_fine = log_euler_path(v0, gammas, kappas, dt, dW_fine)
    v_coarse = log_euler_path(v0, gammas, kappas, 2.0 * dt, dW_coarse)
    return v_fine[2::2], v_coarse[1:], Z


def l2_error(v_ref: np.ndarray, v_approx: np.ndarray) -> float:
    """Relative L2 distance between two path ensembles on the same time grid."""
    num = float(np.sqrt(np.mean((v_ref - v_approx) ** 2)))
    den = float(np.sqrt(np.mean(v_ref ** 2)))
    return num / den if den > 0 else float("nan")


def convergence_table(beta: float, gamma: float, kappa: float, v0: float = 1.0,
                      T: float = 20.0, dt0: float = 0.01, n_paths: int = 200,
                      seed: int = 918273, n_halvings: int = 3):
    """Refine dt by successive halvings, each compared against its own half-dt solution.

    For every level ``k`` the reference is the solution at ``dt/2`` driven by the *same*
    Brownian path, so the reported error is a discretisation bias, not a sampling difference.
    """
    rows = []
    for k in range(n_halvings + 1):
        dt = dt0 / (2 ** k)
        n_fine = int(round(T / dt))
        v_fine, v_coarse, _Z = crm_pair(v0, (gamma,), (kappa,), dt, n_fine, n_paths, seed)
        rows.append(dict(level=k, dt=dt, n_steps=n_fine,
                         l2_rel=l2_error(v_fine, v_coarse),
                         n_paths=n_paths, T=T, seed=seed))
    # observed order from consecutive levels
    for k in range(len(rows) - 1):
        e0, e1 = rows[k]["l2_rel"], rows[k + 1]["l2_rel"]
        rows[k]["order_vs_next"] = (math.log(e0 / e1) / math.log(2.0)
                                    if e0 > 0 and e1 > 0 else float("nan"))
    rows[-1]["order_vs_next"] = float("nan")
    return rows


__all__ = ["convergence_table", "crm_pair", "l2_error", "log_euler_path"]
