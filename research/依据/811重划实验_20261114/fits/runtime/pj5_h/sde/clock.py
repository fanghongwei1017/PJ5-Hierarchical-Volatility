"""Trading-clock extension of the hierarchical background SDE (the T06 original part).

Model (docs/theory/novelty_boundary.txt Eq. E1, PIPELINE lines 168-174)::

    a_chi(t) = s(t) + chi * (1 - s(t)),      chi in [0, 1]
    dv_i     = a_chi(t) * gamma_i * (v_{i-1} - v_i) dt
               + sqrt(a_chi(t)) * kappa_i * v_i dW_i
    dx       = -a_chi(t) mu x dt + sqrt(a_chi(t)) * sqrt(c mu v) dW

with ``s(t) = 1`` inside a trading session and 0 in the lunch / overnight gaps.  The same
``a`` multiplies the drift of every level AND sits under the square root of the diffusion --
that is the whole point of the construction and the one thing an implementation can get
wrong (see ``audit/evidence/T06/``).  Returns are observed only while ``s(t) = 1``.

Time units
----------
Everything is expressed in units of the *slowest* relaxation time ``tau_1 = 1/gamma_1``.
The dimensionless gap length is then

    g = gamma_1 * Delta_gap = Delta_gap / tau_1,

i.e. *g is literally the overnight gap measured in units of tau_1*: g = 1 means the slow
background relaxes once over one overnight gap, g = 5 means it decorrelates fully, g = 0.25
means it is nearly frozen.  This makes g the only knob and keeps gamma_1 fixed, as the design
matrix requires ("keep gamma_N fixed, only move the upper rates").

Real A-share geometry (verified below, minutes)
-----------------------------------------------
morning 09:30-11:30 = 120, lunch 11:30-13:00 = 90, afternoon 13:00-15:00 = 120,
overnight 15:00 -> next 09:30 = 1110 (= 18.5 h), day = 1440.

NOTE: the task statement says "overnight 17.5 h".  15:00 to 09:30 is **18.5 h = 1110 min**,
and 17.5 h would require a 08:30 open.  The geometry below is built from the frozen session
rule in ``configs/contracts/interfaces.json`` (``A_share_SH_SZ``: morning 09:30-11:30,
afternoon 13:00-15:00) and is checked in ``tests/sde/test_clock.py``.  See
``audit/evidence/T06/clock_geometry.md``.

Segment lengths are integer multiples of one bar: session = 24 bars, lunch = 18 bars
(=90/5), overnight = 222 bars (=1110/5).  With ``sub_steps`` steps per bar that is
``96 / 72 / 888`` steps at ``sub_steps = 4`` and one day = ``1152`` steps for every g --
only ``dt = g * (5/1110) / sub_steps`` scales with g.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MINUTES_PER_DAY = 1440.0
MORNING_OPEN_MIN = 9 * 60 + 30          # 09:30
MORNING_CLOSE_MIN = 11 * 60 + 30        # 11:30
AFTERNOON_OPEN_MIN = 13 * 60            # 13:00
AFTERNOON_CLOSE_MIN = 15 * 60           # 15:00

SESSION_MINUTES = (MORNING_CLOSE_MIN - MORNING_OPEN_MIN,
                   AFTERNOON_CLOSE_MIN - AFTERNOON_OPEN_MIN)      # (120, 120)
LUNCH_GAP_MIN = AFTERNOON_OPEN_MIN - MORNING_CLOSE_MIN            # 90
OVERNIGHT_GAP_MIN = (MINUTES_PER_DAY - AFTERNOON_CLOSE_MIN) + MORNING_OPEN_MIN  # 1110
DAY_MIN = sum(SESSION_MINUTES) + LUNCH_GAP_MIN + OVERNIGHT_GAP_MIN             # 1440

BARS_PER_SESSION = 24          # 5-minute bars
BAR_MINUTES = SESSION_MINUTES[0] / BARS_PER_SESSION                # 5.0
BARS_PER_DAY = 2 * BARS_PER_SESSION                                # 48
LUNCH_BARS = LUNCH_GAP_MIN / BAR_MINUTES                           # 18.0
OVERNIGHT_BARS = OVERNIGHT_GAP_MIN / BAR_MINUTES                   # 222.0

#: ratio of every calendar length to the overnight gap; only g (the gap in tau_1 units)
#: multiplies these.
REL_SESSION = SESSION_MINUTES[0] / OVERNIGHT_GAP_MIN                # 120/1110
REL_LUNCH = LUNCH_GAP_MIN / OVERNIGHT_GAP_MIN                       # 90/1110
REL_OVERNIGHT = 1.0
REL_BAR = BAR_MINUTES / OVERNIGHT_GAP_MIN                           # 5/1110

GAP_NONE, GAP_LUNCH, GAP_OVERNIGHT = 0, 1, 2


def geometry_report() -> dict:
    """Machine-checkable statement of the calendar assumptions used here."""
    return {
        "morning": [MORNING_OPEN_MIN, MORNING_CLOSE_MIN],
        "afternoon": [AFTERNOON_OPEN_MIN, AFTERNOON_CLOSE_MIN],
        "session_minutes": list(SESSION_MINUTES),
        "lunch_gap_minutes": LUNCH_GAP_MIN,
        "overnight_gap_minutes": OVERNIGHT_GAP_MIN,
        "overnight_gap_hours": OVERNIGHT_GAP_MIN / 60.0,
        "day_minutes": DAY_MIN,
        "bars_per_session": BARS_PER_SESSION,
        "bar_minutes": BAR_MINUTES,
        "open_fraction_of_day": sum(SESSION_MINUTES) / DAY_MIN,
        "task_statement_overnight_hours": 17.5,
        "task_statement_vs_verified": (
            "17.5 h is inconsistent with a 09:30 open and a 15:00 close; "
            "15:00 -> 09:30 is 18.5 h = 1110 min. 1110 min is used."
        ),
    }


@dataclass(slots=True)
class TradingClock:
    """Discrete step layout of one ``(g, chi)`` cell.

    Attributes are flat arrays of length ``n_steps`` so that the integrator's inner loop has
    no per-step branching beyond integer lookups.
    """

    g: float
    chi: float
    sub_steps: int
    n_days: int
    burn_days: int
    gaps_on: bool

    a: np.ndarray = field(repr=False)
    is_open: np.ndarray = field(repr=False)
    bar_of_step: np.ndarray = field(repr=False)
    is_bar_end: np.ndarray = field(repr=False)
    bar_day: np.ndarray = field(repr=False)
    bar_slot: np.ndarray = field(repr=False)
    bar_gap_before: np.ndarray = field(repr=False)
    burn_step: int = 0

    # ---- derived scalars -----------------------------------------------------------------
    @property
    def n_steps(self) -> int:
        return int(self.a.size)

    @property
    def n_bars(self) -> int:
        return int(self.bar_day.size)

    @property
    def dt(self) -> float:
        """Step in units of tau_1 = 1/gamma_1."""
        return self.g * REL_BAR / self.sub_steps

    @property
    def session_units(self) -> float:
        return self.g * REL_SESSION

    @property
    def lunch_units(self) -> float:
        return self.g * REL_LUNCH if self.gaps_on else 0.0

    @property
    def overnight_units(self) -> float:
        return self.g * REL_OVERNIGHT if self.gaps_on else 0.0

    @property
    def total_units(self) -> float:
        return self.n_steps * self.dt

    @property
    def activity_fraction(self) -> float:
        return float(self.a.mean())

    def summary(self) -> dict:
        return {
            "g": self.g,
            "chi": self.chi,
            "gaps_on": self.gaps_on,
            "sub_steps": self.sub_steps,
            "n_days": self.n_days,
            "burn_days": self.burn_days,
            "n_steps": self.n_steps,
            "burn_step": self.burn_step,
            "n_bars": self.n_bars,
            "dt_tau1": self.dt,
            "session_tau1": self.session_units,
            "lunch_tau1": self.lunch_units,
            "overnight_tau1": self.overnight_units,
            "total_tau1": self.total_units,
            "open_fraction": self.activity_fraction,
            "steps_per_day": int(round(self.n_steps / max(1, self.n_days))),
        }


def choose_n_days(g: float, chi: float, target_relaxations: float = 30.0,
                  n_min: int = 40, n_max: int = 120) -> int:
    """Days needed so that the *effective* (activity-weighted) time covers ``target_relaxations``.

    With a_chi the slow layer only ages while the market is open (chi = 0) or partly while it
    is closed (0 < chi < 1); using the calendar day count alone would silently compare cells
    with wildly different effective horizons.
    """
    per_day = g * (REL_SESSION * 2.0 + chi * (REL_LUNCH + REL_OVERNIGHT))
    if per_day <= 0:
        return n_min
    n = int(math.ceil(target_relaxations / per_day))
    return int(min(n_max, max(n_min, n)))


def build_clock(g: float, chi: float, sub_steps: int = 12, n_days: int | None = None,
                burn_fraction: float = 0.25, gaps_on: bool = True,
                n_min: int = 40, n_max: int = 120) -> TradingClock:
    """Build the step layout.

    ``gaps_on = False`` gives the zero-gap control: identical bar and session lengths, no
    closed periods at all.  It is the ``g = 0`` row of the design -- the literal g = 0 makes
    every calendar length vanish (Delta_gap = 0 scales the whole day), which is not
    simulable, so the zero-gap control is run at the *open-time horizon* of the smallest
    simulated g instead.  This deviation is recorded in the audit and the handoff.
    """
    if g <= 0 and gaps_on:
        raise ValueError("g must be > 0 when gaps are enabled; use gaps_on=False for the "
                         "zero-gap control (a literal g = 0 collapses every calendar length)")
    if not (0.0 <= chi <= 1.0):
        raise ValueError("chi must lie in [0, 1]")
    if sub_steps < 1:
        raise ValueError("sub_steps must be >= 1")

    if n_days is None:
        n_days = choose_n_days(g, chi, n_min=n_min, n_max=n_max)
    burn_days = int(round(burn_fraction * n_days))

    sess_steps = BARS_PER_SESSION * sub_steps
    lunch_steps = int(round(LUNCH_BARS * sub_steps)) if gaps_on else 0
    night_steps = int(round(OVERNIGHT_BARS * sub_steps)) if gaps_on else 0
    steps_per_day = 2 * sess_steps + lunch_steps + night_steps

    n_steps = n_days * steps_per_day
    a = np.empty(n_steps)
    is_open = np.zeros(n_steps, dtype=bool)
    bar_of_step = np.full(n_steps, -1, dtype=np.int32)
    is_bar_end = np.zeros(n_steps, dtype=bool)

    n_bars = n_days * BARS_PER_DAY
    bar_day = np.empty(n_bars, dtype=np.int32)
    bar_slot = np.empty(n_bars, dtype=np.int32)
    bar_gap_before = np.zeros(n_bars, dtype=np.int8)

    k = 0
    b = 0
    for d in range(n_days):
        for s in range(2):
            # gap before this session
            kind = GAP_NONE
            if s == 1 and gaps_on:
                a[k:k + lunch_steps] = chi
                k += lunch_steps
                kind = GAP_LUNCH
            for j in range(BARS_PER_SESSION):
                bid = b
                bar_day[bid] = d
                bar_slot[bid] = s * BARS_PER_SESSION + j
                bar_gap_before[bid] = kind if j == 0 else GAP_NONE
                a[k:k + sub_steps] = 1.0
                is_open[k:k + sub_steps] = True
                bar_of_step[k:k + sub_steps] = bid
                is_bar_end[k + sub_steps - 1] = True
                k += sub_steps
                b += 1
        if gaps_on:
            a[k:k + night_steps] = chi
            k += night_steps
            # the first bar of the next day carries GAP_OVERNIGHT; set below
    if k != n_steps:                                        # pragma: no cover - structural
        raise AssertionError(f"step layout mismatch: filled {k} of {n_steps}")
    if b != n_bars:                                         # pragma: no cover - structural
        raise AssertionError(f"bar layout mismatch: filled {b} of {n_bars}")

    # first bar of each day (except the first) is preceded by the overnight gap
    if gaps_on:
        for d in range(1, n_days):
            bar_gap_before[d * BARS_PER_DAY] = GAP_OVERNIGHT

    return TradingClock(
        g=float(g), chi=float(chi), sub_steps=int(sub_steps), n_days=int(n_days),
        burn_days=burn_days, gaps_on=bool(gaps_on),
        a=a, is_open=is_open, bar_of_step=bar_of_step, is_bar_end=is_bar_end,
        bar_day=bar_day, bar_slot=bar_slot, bar_gap_before=bar_gap_before,
        burn_step=int(burn_days * steps_per_day),
    )


def continuous_activity(n_steps: int) -> np.ndarray:
    """chi = 1 with no gaps: the plain 24/7 clock of the paper."""
    return np.ones(int(n_steps))


__all__ = [
    "AFTERNOON_CLOSE_MIN",
    "AFTERNOON_OPEN_MIN",
    "BAR_MINUTES",
    "BARS_PER_DAY",
    "BARS_PER_SESSION",
    "DAY_MIN",
    "GAP_LUNCH",
    "GAP_NONE",
    "GAP_OVERNIGHT",
    "LUNCH_GAP_MIN",
    "MORNING_CLOSE_MIN",
    "MORNING_OPEN_MIN",
    "OVERNIGHT_GAP_MIN",
    "SESSION_MINUTES",
    "TradingClock",
    "build_clock",
    "choose_n_days",
    "continuous_activity",
    "geometry_report",
]
