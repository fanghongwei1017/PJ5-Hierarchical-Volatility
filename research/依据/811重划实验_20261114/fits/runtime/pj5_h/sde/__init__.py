"""T06: hierarchical background SDE, trading-clock extension, numerical diagnostics.

Public API
----------
    from pj5_h.sde import HierarchyConfig, simulate, simulate_batch
    from pj5_h.sde import metrics, clock

Design rules fixed by the T00 contract (docs/theory/equation_contract.txt):
  * background SDE           dv_i = -gamma_i (v_i - v_{i-1}) dt + kappa_i v_i dW_i
  * conditional stationary   v_i | v_{i-1} = u  ~  InvGamma(beta+1, beta u)
  * integration              log variable y_i = ln v_i, Ito correction -kappa_i^2/2
  * NO abs(v), NO clipping of negatives, NO silent truncation anywhere
  * two return modes, always declared:
        paper_literal       dx = -mu x dt + sqrt(mu v)   dW  -> stationary var v/2
        variance_consistent dx = -mu x dt + sqrt(2 mu v) dW  -> stationary var v
"""

from __future__ import annotations

from .integrator import HierarchyConfig, simulate_hierarchy
from .simulate import simulate, simulate_batch

__all__ = [
    "HierarchyConfig",
    "simulate",
    "simulate_batch",
    "simulate_hierarchy",
]
