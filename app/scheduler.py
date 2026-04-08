"""
Rules-based scheduling engine (Step 5).

Reads category time windows, keyword overrides, and duration defaults
from prompts/rules.py. No Google Calendar calls here — this module
finds slots against a list of busy (start, end) datetime tuples,
which the caller populates from the freebusy API (Step 6).

Key constraint: tasks are always scheduled on their written day.
A Wednesday task never moves to Monday even if Monday has open slots.
"""

import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

from app.models import Priority

# Load rules from prompts/rules.py without making it an importable package
_rules_path = Path(__file__).parent.parent / "prompts" / "rules.py"
_rules_ns: dict = {}
exec(_rules_path.read_text(), _rules_ns)
RULES: dict = _rules_ns["SCHEDULING_RULES"]

# Day abbreviation used in rules.py
_WEEKDAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _parse_time(t: str) -> time:
    h, m = t.split(":")
    return time(int(h), int(m))


def resolve_category(task: Priority) -> str:
    """
    Return the effective scheduling category for a task.
    Keyword overrides take precedence over category_hint.
    """
    text_lower = task.text.lower()
    for keyword, category in RULES["keyword_overrides"].items():
        if keyword in text_lower:
            return category
    return task.category_hint


def _get_window(category: str, target_date: date) -> tuple[datetime, datetime] | None:
    """
    Return the (window_start, window_end) datetimes for a category on a given date.
    Returns None if the category cannot be scheduled on that day.
    """
    day_abbr = _WEEKDAY_ABBR[target_date.weekday()]

    # Special case: errands on Sunday use the preferred shopping window
    if category == "errand" and day_abbr == "Sun":
        win_start = _parse_time(RULES["preferred_shopping_window"]["start"])
        win_end = _parse_time(RULES["preferred_shopping_window"]["end"])
        return (
            datetime.combine(target_date, win_start),
            datetime.combine(target_date, win_end),
        )

    window_config = RULES["category_windows"].get(category)
    if window_config is None or day_abbr not in window_config["days"]:
        return None

    hard_start = _parse_time(RULES["hard_start"])
    hard_end = _parse_time(RULES["hard_end"])

    win_start = max(_parse_time(window_config["start"]), hard_start)
    win_end = min(_parse_time(window_config["end"]), hard_end)

    return (
        datetime.combine(target_date, win_start),
        datetime.combine(target_date, win_end),
    )


def find_slot(
    task: Priority,
    target_date: date,
    busy_slots: list[tuple[datetime, datetime]],
) -> tuple[datetime, datetime] | None:
    """
    Find the first available slot for a task on its target date.

    Returns (start, end) or None if no slot fits within the category window.
    Busy slots are sorted and scanned left-to-right; buffer_minutes gap is
    enforced after each busy period.
    """
    category = resolve_category(task)
    window = _get_window(category, target_date)
    if window is None:
        return None

    win_start, win_end = window
    duration = timedelta(minutes=RULES["default_durations"].get(category, 30))
    buffer = timedelta(minutes=RULES["buffer_minutes"])

    # Only consider busy slots that overlap with our window
    relevant = sorted(
        [(s, e) for s, e in busy_slots if s < win_end and e > win_start],
        key=lambda x: x[0],
    )

    candidate = win_start
    for busy_start, busy_end in relevant:
        if candidate + duration <= busy_start:
            break  # Gap before this busy block is large enough
        # Move past the busy block + buffer
        candidate = max(candidate, busy_end + buffer)

    if candidate + duration <= win_end:
        return (candidate, candidate + duration)

    return None


def schedule_day(
    tasks: list[Priority],
    target_date: date,
    busy_slots: list[tuple[datetime, datetime]],
) -> list[Priority]:
    """
    Schedule all tasks for a single day against a list of busy slots.

    Tasks are processed in position order. Each successfully scheduled task
    is added to busy_slots so subsequent tasks avoid it. Tasks that cannot
    be scheduled (wrong day, window full) are returned with scheduled_start=None.

    Returns a new list of Priority objects — originals are not mutated.
    """
    accumulated_busy = list(busy_slots)
    result = []

    for task in sorted(tasks, key=lambda t: t.position):
        slot = find_slot(task, target_date, accumulated_busy)
        if slot:
            start, end = slot
            scheduled = task.model_copy(update={
                "scheduled_start": start,
                "scheduled_end": end,
            })
            accumulated_busy.append(slot)
        else:
            scheduled = task.model_copy(update={
                "scheduled_start": None,
                "scheduled_end": None,
            })
        result.append(scheduled)

    return result


def schedule_week(plan: "WeeklyPlan") -> "WeeklyPlan":
    """
    Schedule all tasks in a WeeklyPlan against Google Calendar.

    For each day:
    1. Fetch busy slots from GCal freebusy
    2. Run rules engine (schedule_day)
    3. For tasks the rules engine couldn't place, try conflict resolution LLM
    4. Create calendar events for all successfully scheduled tasks

    Returns an updated WeeklyPlan with scheduled_start, scheduled_end,
    and calendar_event_id set on each placed task.
    """
    from app.models import WeeklyPlan
    from app.gcal import create_calendar_event, get_busy_slots, resolve_conflict

    updated_days = []
    for day in plan.days:
        if not day.priorities:
            updated_days.append(day)
            continue

        busy_slots = get_busy_slots(day.date)
        scheduled_tasks = schedule_day(day.priorities, day.date, busy_slots)

        final_tasks = []
        for task in scheduled_tasks:
            if task.scheduled_start is None:
                # Rules engine found no slot — try LLM fallback
                slot = resolve_conflict(task, day.date, open_slots=_free_slots(busy_slots, day.date))
                if slot:
                    task = task.model_copy(update={
                        "scheduled_start": slot[0],
                        "scheduled_end": slot[1],
                    })

            if task.scheduled_start is not None:
                try:
                    event_id = create_calendar_event(task)
                    task = task.model_copy(update={"calendar_event_id": event_id})
                except Exception:
                    pass  # Event creation failure doesn't block the rest

            final_tasks.append(task)

        updated_days.append(day.model_copy(update={"priorities": final_tasks}))

    return plan.model_copy(update={"days": updated_days})


def _free_slots(
    busy_slots: list[tuple],
    target_date: "date",
    window_start_h: int = 6,
    window_end_h: int = 21,
) -> list[tuple]:
    """Return free intervals in the hard day window given busy slots."""
    from datetime import datetime as dt
    day_start = dt(target_date.year, target_date.month, target_date.day, window_start_h, 0)
    day_end = dt(target_date.year, target_date.month, target_date.day, window_end_h, 0)

    sorted_busy = sorted(busy_slots, key=lambda s: s[0])
    free = []
    cursor = day_start
    for bstart, bend in sorted_busy:
        if cursor < bstart:
            free.append((cursor, bstart))
        cursor = max(cursor, bend)
    if cursor < day_end:
        free.append((cursor, day_end))
    return free
