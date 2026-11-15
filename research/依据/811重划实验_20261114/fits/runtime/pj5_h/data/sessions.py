"""A-share (Shanghai / Shenzhen) trading-session rules and the intraday label grid.

Everything in this module is *static structure*, not estimated from data.  It encodes
the frozen facts of ``docs/data/data_contract.txt`` (§3, §4) and
``configs/contracts/interfaces.json`` (``modules.data.session_rules``):

* one trading day has two continuous sessions, ``09:30-11:30`` (session_id 1, "am") and
  ``13:00-15:00`` (session_id 2, "pm"), 120 minutes each, 240 minutes per day;
* the vendor timestamp is the **left endpoint** of the bar, so the 1-minute labels of a
  full day are ``09:30..11:29`` (120) plus ``13:00..14:59`` (120), i.e. the intervals
  ``[09:30,09:31) ... [11:29,11:30)`` and ``[13:00,13:01) ... [14:59,15:00)``;
* ``11:30`` and ``15:00`` are **terminal prints**, not bars (§3.1, §3.3, §3.4).  The
  ``15:00`` row carries the official close price (§3.4a) and its volume belongs to the
  day (§3.4b); the ``11:30`` row is a tiny extra print (§3.5).  Both are merged into the
  *volume* of the last bar of their session, and the ``15:00`` row additionally replaces
  the *close* of the last afternoon bar.  See :mod:`pj5_h.data.build`.
* ``14:57``, ``14:58``, ``14:59`` are **structurally absent** from the vendor feed after
  2018 (missing in 35.4 %, 49.9 %, 49.7 % of all files, versus < 12 % for any other
  label; §3.6, §10.1).  They are therefore excluded from the *required* label set when
  the 80 % label-coverage rule of :func:`pj5_h.data.build.build_returns` is evaluated,
  but they are still used when present.  This is a property of the vendor feed, decided
  in T01 from whole-sample *format* evidence -- it carries no return information and is
  not refitted downstream.

All times are handled internally as *minutes of the day* (integer, 0..1439, naive
Asia/Shanghai wall clock -- the data contract forbids mixing in UTC).  Conversion helpers
are provided at the bottom of the module.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

__all__ = [
    "SESSION_RULES",
    "SESSION_RULES_ID",
    "STRUCTURAL_ABSENT_LABELS",
    "TERMINAL_MINUTES",
    "minutes_of_day",
    "minutes_to_hhmmss",
    "minutes_to_label",
    "session_ids",
    "session_bounds",
    "session_minutes",
    "session_of_minutes",
    "is_terminal_minutes",
    "interval_count",
    "interval_starts",
    "interval_label_minutes",
    "interval_required_minutes",
    "interval_is_session_last",
    "n_slots",
    "slot_index",
    "slot_tod_label",
    "resolve_rules",
    "label_grid",
]

# --------------------------------------------------------------------------- constants
SESSION_RULES_ID = "A_share_SH_SZ"

#: minutes of the day: 09:30 = 570, 11:30 = 690, 13:00 = 780, 15:00 = 900
_AM_START, _AM_END = 9 * 60 + 30, 11 * 60 + 30
_PM_START, _PM_END = 13 * 60, 15 * 60

#: The three closing labels the vendor drops systematically from 2018 on (contract §3.6).
STRUCTURAL_ABSENT_LABELS: Tuple[int, ...] = (14 * 60 + 57, 14 * 60 + 58, 14 * 60 + 59)

#: Labels that are terminal prints rather than bars (contract §3.1).
TERMINAL_MINUTES: Tuple[int, ...] = (_AM_END, _PM_END)

SESSION_RULES: Dict = {
    "id": SESSION_RULES_ID,
    "morning": ["09:30", "11:30"],
    "afternoon": ["13:00", "15:00"],
    "minutes_per_day": 240,
    "timestamp_semantics": (
        "bar left-endpoint label; the 09:30 bar already contains the opening auction; "
        "11:30 and 15:00 are terminal prints, 15:00 carries the official close"
    ),
    "sessions": [
        {
            "session_id": 1,
            "name": "am",
            "start_min": _AM_START,
            "end_min": _AM_END,
            "minutes": _AM_END - _AM_START,
            "terminal_min": _AM_END,
        },
        {
            "session_id": 2,
            "name": "pm",
            "start_min": _PM_START,
            "end_min": _PM_END,
            "minutes": _PM_END - _PM_START,
            "terminal_min": _PM_END,
        },
    ],
    "structural_absent_minutes": list(STRUCTURAL_ABSENT_LABELS),
    "terminal_minutes": list(TERMINAL_MINUTES),
}


# --------------------------------------------------------------------------- helpers
def minutes_of_day(hhmmssmmm: int) -> Tuple[int, int, int]:
    """Split the vendor integer timestamp ``HHMMSSmmm`` into (minutes, seconds, millis).

    >>> minutes_of_day(93000000)
    (570, 0, 0)
    >>> minutes_of_day(145959123)
    (899, 59, 123)
    """
    t = int(hhmmssmmm)
    if t < 0:
        raise ValueError("negative timestamp: %r" % (hhmmssmmm,))
    hh, rest = divmod(t, 10_000_000)
    mm, rest = divmod(rest, 100_000)
    ss, ms = divmod(rest, 1_000)
    if hh > 23 or mm > 59 or ss > 59:
        raise ValueError("not an HHMMSSmmm timestamp: %r" % (hhmmssmmm,))
    return hh * 60 + mm, ss, ms


def minutes_to_hhmmss(minute: int) -> str:
    """``570 -> '09:30:00'`` (seconds are always 00 on this grid)."""
    minute = int(minute)
    hh, mm = divmod(minute, 60)
    return "%02d:%02d:00" % (hh, mm)


def minutes_to_label(minute: int) -> str:
    """``570 -> '09:30'`` -- the short form used as a time-of-day slot key."""
    minute = int(minute)
    hh, mm = divmod(minute, 60)
    return "%02d:%02d" % (hh, mm)


def session_ids() -> Tuple[int, ...]:
    return tuple(s["session_id"] for s in SESSION_RULES["sessions"])


def session_bounds(session_id: int) -> Tuple[int, int]:
    """(start_min, end_min) of one session in minutes of the day."""
    for s in SESSION_RULES["sessions"]:
        if s["session_id"] == int(session_id):
            return s["start_min"], s["end_min"]
    raise KeyError("unknown session_id %r" % (session_id,))


def session_minutes(session_id: int) -> int:
    a, b = session_bounds(session_id)
    return b - a


def session_of_minutes(minute: int) -> int:
    """Session id of a grid label; terminal labels belong to the session they close."""
    m = int(minute)
    if _AM_START <= m <= _AM_END:
        return 1
    if _PM_START <= m <= _PM_END:
        return 2
    raise ValueError("minute %d (%s) is outside both sessions" % (m, minutes_to_hhmmss(m)))


def is_terminal_minutes(minute: int) -> bool:
    return int(minute) in TERMINAL_MINUTES


# --------------------------------------------------------------------------- grid
def interval_count(frequency: int, session_id: int) -> int:
    """Number of k-minute intervals in one session (120 / k)."""
    k = int(frequency)
    if k <= 0 or 120 % k:
        raise ValueError("frequency must divide 120 minutes: %r" % (frequency,))
    return session_minutes(session_id) // k


def interval_starts(frequency: int, session_id: int) -> List[int]:
    """Left endpoints (minutes of the day) of the k-minute intervals of one session."""
    k = int(frequency)
    start, _ = session_bounds(session_id)
    return [start + k * j for j in range(interval_count(k, session_id))]


def interval_label_minutes(frequency: int, start_min: int) -> List[int]:
    """The k one-minute grid labels covered by the interval ``[start_min, start_min+k)``."""
    k = int(frequency)
    return [int(start_min) + i for i in range(k)]


def interval_required_minutes(frequency: int, start_min: int) -> List[int]:
    """Labels that *must* be present for the interval to be usable.

    Drops the structurally absent closing labels (see module docstring); if that empties
    the interval (only possible for k = 1 on 14:57/14:58/14:59) the caller should treat
    the interval as optional: build it when the label happens to exist, skip it silently
    otherwise.
    """
    return [m for m in interval_label_minutes(frequency, start_min)
            if m not in STRUCTURAL_ABSENT_LABELS]


def interval_is_session_last(frequency: int, start_min: int) -> bool:
    sid = session_of_minutes(int(start_min))
    _, end = session_bounds(sid)
    return int(start_min) + int(frequency) >= end


def n_slots(frequency: int) -> int:
    """Number of time-of-day slots of one trading day at this frequency (am + pm)."""
    return sum(interval_count(frequency, sid) for sid in session_ids())


def slot_index(frequency: int, session_id: int, start_min: int) -> int:
    """0-based slot index over the whole day: am slots first, then pm slots."""
    k = int(frequency)
    off = 0
    for sid in session_ids():
        if sid == int(session_id):
            break
        off += interval_count(k, sid)
    else:  # pragma: no cover - unreachable for valid session ids
        raise KeyError(session_id)
    start, _ = session_bounds(session_id)
    return off + (int(start_min) - start) // k


def slot_tod_label(start_min: int) -> str:
    return minutes_to_label(start_min)


def label_grid(frequency: int) -> List[int]:
    """All k-minute interval left endpoints of one day, am first then pm."""
    out: List[int] = []
    for sid in session_ids():
        out.extend(interval_starts(frequency, sid))
    return out


def resolve_rules(session_rules) -> Dict:
    """Accept the rules dict (or its id) and validate the frozen A-share structure."""
    if session_rules is None:
        return SESSION_RULES
    if isinstance(session_rules, str):
        if session_rules != SESSION_RULES_ID:
            raise KeyError("unknown session rule set %r (only %r is frozen)"
                           % (session_rules, SESSION_RULES_ID))
        return SESSION_RULES
    if not isinstance(session_rules, dict):
        raise TypeError("session_rules must be a dict or %r" % (SESSION_RULES_ID,))
    rules = dict(SESSION_RULES)
    rules.update(session_rules)
    if rules.get("minutes_per_day") != 240:
        raise ValueError("this implementation only supports the frozen 240-minute A-share day")
    return rules
