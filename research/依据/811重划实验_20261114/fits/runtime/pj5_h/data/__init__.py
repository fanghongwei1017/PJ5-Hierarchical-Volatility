"""Minute-data layer of the PJ5 pipeline (T03).

Public API (frozen signatures in ``configs/contracts/interfaces.json``)::

    from pj5_h.data.bars import load_day_symbol          # vendor CSV -> bar table
    from pj5_h.data.build import build_returns           # bars -> (returns, exclusions)
    from pj5_h.data.seasonality import fit_seasonality   # train-only s(tod)
    from pj5_h.data.sessions import SESSION_RULES        # A-share session grid

Everything downstream (H-theory density fits, model selection) consumes only the
``returns`` tables produced here; nothing in this package fits a model.
"""

from __future__ import annotations

from . import bars, build, manifest, seasonality, sessions  # noqa: F401
from .bars import load_day_symbol  # noqa: F401
from .build import build_returns, collect_gaps  # noqa: F401
from .seasonality import apply_seasonality, fit_seasonality  # noqa: F401
from .sessions import SESSION_RULES, SESSION_RULES_ID  # noqa: F401

__all__ = [
    "bars",
    "build",
    "manifest",
    "seasonality",
    "sessions",
    "load_day_symbol",
    "build_returns",
    "collect_gaps",
    "fit_seasonality",
    "apply_seasonality",
    "SESSION_RULES",
    "SESSION_RULES_ID",
]
