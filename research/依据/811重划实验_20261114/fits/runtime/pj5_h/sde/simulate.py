"""``sde.simulate`` -- the frozen T00 interface, plus batch helpers.

Interface (``configs/contracts/interfaces.json``)::

    simulate(config, seed) -> dict(latent_paths, returns, diagnostics)

``config_required``: N, beta, v0, r, gamma_N, mu_over_gamma_N, dt, t_total, burn_in,
``ou_mode``, chi, g_gap, session_pattern.

Two *structural* additions beyond that list, both explicit rather than hidden:

``n_paths``    how many independent paths share the seed (the Monte-Carlo size)
``n_record``   how many time slices of the post-burn-in window are stored

Neither changes the model; both are recorded in ``diagnostics`` and in the run config.
"""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np

from .integrator import HierarchyConfig, simulate_hierarchy

_FIELD_NAMES = {f.name for f in fields(HierarchyConfig)}

#: defaults used when a config omits a required key but the value is unambiguous
_DEFAULTS = {
    "v0": 1.0,
    "r": 1.0,
    "gamma_N": 1.0,
    "mu_over_gamma_N": 10.0,
    "ou_mode": "variance_consistent",
    "chi": 1.0,
    "g_gap": 0.0,
    "session_pattern": "continuous",
}

REQUIRED = ("N", "beta", "v0", "r", "gamma_N", "mu_over_gamma_N", "dt", "t_total",
            "burn_in", "ou_mode", "chi", "g_gap", "session_pattern")


def config_from_dict(d: dict) -> HierarchyConfig:
    """Build a :class:`HierarchyConfig` from a plain dict (contract config or JSON file).

    Unknown keys raise: silently dropping a misspelled knob is exactly the kind of quiet
    failure the contract forbids.
    """
    unknown = sorted(set(d) - _FIELD_NAMES)
    if unknown:
        raise KeyError(f"unknown config keys: {unknown}; known keys are {sorted(_FIELD_NAMES)}")
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise KeyError(f"config is missing required keys {missing}")
    merged = dict(_DEFAULTS)
    merged.update({k: v for k, v in d.items() if v is not None})
    return HierarchyConfig(**merged)


def simulate(config, seed: int) -> dict:
    """Run one simulation; ``config`` may be a dict or a :class:`HierarchyConfig`."""
    cfg = config if isinstance(config, HierarchyConfig) else config_from_dict(config)
    return simulate_hierarchy(cfg, seed)


def simulate_batch(config, seeds) -> list:
    """Same config, several seeds.  Seeds are *numerical replicates*, never new markets."""
    return [simulate(config, s) for s in seeds]


# ---------------------------------------------------------------------------------------
# horizon helpers -- the design requires burn-in and duration to follow the SLOWEST level
# ---------------------------------------------------------------------------------------


def horizons_for_slowest(gamma_1: float, relax_total: float = 40.0,
                         relax_burn: float = 10.0) -> dict:
    """``t_total`` and ``burn_in`` such that the slowest level relaxes enough times.

    PIPELINE lines 176-179: burn-in and duration are set by the slowest layer, the step by the
    fastest one.  Fixed step counts across different r would compare different effective
    simulation lengths, so this helper is always used instead.
    """
    return {"burn_in": relax_burn / gamma_1, "t_total": relax_total / gamma_1,
            "slowest_relaxation_units": relax_total, "burn_relaxation_units": relax_burn}


def dt_for_fastest(gamma_N: float, steps_per_relaxation: float = 100.0) -> float:
    """Step size resolving the fastest relaxation time ``1/gamma_N``."""
    return 1.0 / (gamma_N * steps_per_relaxation)


def stack_top_level(runs) -> np.ndarray:
    """Concatenate the top-level recorded background across seeds -> (seeds*P, n_rec)."""
    return np.vstack([np.asarray(r["latent_paths"]["vN"]) for r in runs])


def stack_returns(runs) -> np.ndarray:
    return np.vstack([np.asarray(r["returns"]) for r in runs])


def stack_level(runs, level: int) -> np.ndarray:
    """Concatenate level ``level`` (0-based) of the recorded background across seeds."""
    return np.vstack([np.asarray(r["latent_paths"]["v"])[:, level, :] for r in runs])


__all__ = [
    "REQUIRED",
    "config_from_dict",
    "dt_for_fastest",
    "horizons_for_slowest",
    "simulate",
    "simulate_batch",
    "stack_level",
    "stack_returns",
    "stack_top_level",
]
