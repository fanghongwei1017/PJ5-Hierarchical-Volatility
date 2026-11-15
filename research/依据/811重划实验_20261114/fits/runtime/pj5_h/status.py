"""Numerical status codes. Every density evaluation must return one of these.

Silently returning 0 or NaN for a failed evaluation is forbidden by the T00 contract
(docs/theory/equation_contract.txt, rule N2).
"""

OK = "ok"
UNDERFLOW = "underflow"
OVERFLOW = "overflow"
NO_CONVERGENCE = "no_convergence"
INVALID_PARAM = "invalid_param"

FAILED = frozenset({OVERFLOW, NO_CONVERGENCE, INVALID_PARAM})
