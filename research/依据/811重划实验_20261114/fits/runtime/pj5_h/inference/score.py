"""Hold-out scoring and baselines.

The primary score required by PROTOCOL.json is the test-period return log score, compared
across models by a paired block difference.  The paper's Eq. (24) error is a *training*
histogram criterion; it is kept as a secondary/compatibility score and must never be
described as a likelihood.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..models import return_logpdf, return_logpdf_n1_closed


@dataclass
class GaussianModel:
    mean: float
    sd: float

    def logpdf(self, x):
        return stats.norm.logpdf(np.asarray(x, dtype=float), self.mean, self.sd)


@dataclass
class StudentTModel:
    """Variance-normalised Student-t, i.e. N=1 H-theory written in closed form (Eq. 16)."""

    beta: float
    v0: float = 1.0

    def logpdf(self, x):
        return return_logpdf_n1_closed(np.asarray(x, dtype=float), self.beta, self.v0)


@dataclass
class HModel:
    N: int
    beta: float
    v0: float = 1.0

    def logpdf(self, x):
        lp, _ = return_logpdf(np.asarray(x, dtype=float), self.N, self.beta, self.v0)
        return lp


def fit_gaussian(x_train):
    x = np.asarray(x_train, dtype=float)
    return GaussianModel(mean=float(x.mean()), sd=float(x.std(ddof=1)))


def fit_student_t_n1(x_train, v0: float = 1.0):
    """Fit beta of the N=1 model by maximum likelihood on the training set."""
    from scipy.optimize import minimize_scalar

    x = np.asarray(x_train, dtype=float)

    def nll(log_beta):
        return -np.mean(return_logpdf_n1_closed(x, math.exp(log_beta), v0))

    res = minimize_scalar(nll, bounds=(math.log(0.05), math.log(50.0)), method="bounded")
    return StudentTModel(beta=float(math.exp(res.x)), v0=v0)


def score_logpdf(model, x) -> np.ndarray:
    return np.asarray(model.logpdf(np.asarray(x, dtype=float)), dtype=float)


def block_bootstrap_difference(logp_a: np.ndarray, logp_b: np.ndarray,
                               block_length: int, n_boot: int = 200, seed: int = 0):
    """Paired block bootstrap of mean(logp_a - logp_b).

    ``logp_a`` and ``logp_b`` must be aligned point by point (same observations), so the
    difference is paired.  Blocks are contiguous, which preserves the volatility dependence.
    """
    d = np.asarray(logp_a, dtype=float) - np.asarray(logp_b, dtype=float)
    n = d.size
    L = max(1, int(block_length))
    n_blocks = int(math.ceil(n / L))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    starts = np.arange(0, n, L)
    for i in range(n_boot):
        pick = rng.integers(0, len(starts), size=n_blocks)
        idx = np.concatenate([np.arange(starts[j], min(starts[j] + L, n)) for j in pick])
        means[i] = d[idx].mean()
    return {
        "mean_diff": float(d.mean()),
        "se": float(means.std(ddof=1)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "block_length": L,
        "n_boot": int(n_boot),
        "n_points": int(n),
    }


def estimate_block_length(x, max_lag: int = 200) -> int:
    """Integral autocorrelation time of |x|, the natural block length for dependence.

    NOTE: this environment's BLAS is broken -- ``np.dot`` on arrays longer than about 4000
    elements crashes the interpreter (exit -1066598273).  Every inner product in this module
    therefore uses an elementwise product plus ``np.sum``, never ``np.dot``/``@``.
    """
    a = np.abs(np.asarray(x, dtype=float))
    a = a - a.mean()
    n = a.size
    var = float(np.sum(a * a)) / n
    if var <= 0:
        return 1
    tau = 1.0
    for k in range(1, min(max_lag, n // 4)):
        c = float(np.sum(a[:-k] * a[k:])) / (n - k) / var
        if c <= 0:
            break
        tau += 2.0 * c
    return int(max(1, round(tau)))
