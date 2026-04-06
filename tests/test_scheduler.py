"""
Tests for the rules-based scheduling engine (Step 5).
Run with: venv/bin/python -m pytest tests/test_scheduler.py -v

No Google Calendar — all tests use mocked busy slots.
"""

from datetime import date, datetime, timedelta

import pytest

from app.models import Priority
from app.scheduler import find_slot, resolve_category, schedule_day

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MONDAY    = date(2026, 3, 23)  # weekday() == 0
TUESDAY   = date(2026, 3, 24)
WEDNESDAY = date(2026, 3, 25)
SATURDAY  = date(2026, 3, 28)
SUNDAY    = date(2026, 3, 29)


def make_priority(category: str, text: str = "task", status: str = "open") -> Priority:
    return Priority(
        position=1, text=text, confidence="high",
        category_hint=category, status=status,
    )


def busy(date_: date, start_h: int, end_h: int) -> tuple[datetime, datetime]:
    return (
        datetime(date_.year, date_.month, date_.day, start_h, 0),
        datetime(date_.year, date_.month, date_.day, end_h, 0),
    )


# ---------------------------------------------------------------------------
# resolve_category
# ---------------------------------------------------------------------------

class TestResolveCategory:
    def test_returns_category_hint_when_no_keyword_match(self):
        p = make_priority("work", text="quarterly review")
        assert resolve_category(p) == "work"

    def test_keyword_override_gym(self):
        p = make_priority("personal", text="go to the gym")
        assert resolve_category(p) == "health"

    def test_keyword_override_laundry(self):
        p = make_priority("unknown", text="do laundry")
        assert resolve_category(p) == "chore"

    def test_keyword_override_case_insensitive(self):
        p = make_priority("personal", text="LAUNDRY day")
        assert resolve_category(p) == "chore"

    def test_keyword_override_sams(self):
        p = make_priority("unknown", text="Sam's Club run")
        assert resolve_category(p) == "errand"

    def test_no_override_when_no_match(self):
        p = make_priority("health", text="meditate")
        assert resolve_category(p) == "health"


# ---------------------------------------------------------------------------
# find_slot
# ---------------------------------------------------------------------------

class TestFindSlot:
    def test_work_task_monday_no_conflicts(self):
        p = make_priority("work")
        slot = find_slot(p, MONDAY, [])
        assert slot is not None
        start, end = slot
        assert start.date() == MONDAY
        assert start.hour == 9  # work window starts 09:00
        assert (end - start) == timedelta(minutes=60)

    def test_work_task_sunday_returns_none(self):
        # work category only Mon-Fri
        p = make_priority("work")
        assert find_slot(p, SUNDAY, []) is None

    def test_chore_weekday_evening(self):
        p = make_priority("chore")
        slot = find_slot(p, MONDAY, [])
        assert slot is not None
        start, _ = slot
        assert start.hour >= 17  # chore window starts 17:30

    def test_chore_sunday_returns_none(self):
        # chore category only Mon-Sat
        p = make_priority("chore")
        assert find_slot(p, SUNDAY, []) is None

    def test_errand_saturday(self):
        p = make_priority("errand")
        slot = find_slot(p, SATURDAY, [])
        assert slot is not None

    def test_errand_sunday_uses_preferred_shopping_window(self):
        p = make_priority("errand")
        slot = find_slot(p, SUNDAY, [])
        assert slot is not None
        start, _ = slot
        assert start.hour == 9  # preferred shopping window 09:00-12:00

    def test_skips_busy_slot(self):
        p = make_priority("work")
        # Block 09:00-10:00 (work window start)
        conflicts = [busy(MONDAY, 9, 10)]
        slot = find_slot(p, MONDAY, conflicts)
        assert slot is not None
        start, _ = slot
        # Should start at 10:00 + 15 min buffer = 10:15
        assert start >= datetime(MONDAY.year, MONDAY.month, MONDAY.day, 10, 15)

    def test_buffer_between_tasks(self):
        p = make_priority("work")
        # Block 09:00-10:00, next slot should be 10:15 not 10:00
        conflicts = [busy(MONDAY, 9, 10)]
        slot = find_slot(p, MONDAY, conflicts)
        start, _ = slot
        assert start == datetime(MONDAY.year, MONDAY.month, MONDAY.day, 10, 15)

    def test_returns_none_when_window_full(self):
        p = make_priority("chore")
        # Block entire chore window (17:30-21:00)
        conflicts = [busy(MONDAY, 17, 22)]
        assert find_slot(p, MONDAY, conflicts) is None

    def test_health_task_early_morning(self):
        p = make_priority("health")
        slot = find_slot(p, MONDAY, [])
        assert slot is not None
        start, _ = slot
        assert start.hour == 6  # health window starts 06:00

    def test_slot_does_not_exceed_hard_end(self):
        p = make_priority("unknown")
        # Block most of the unknown window, leaving only 30 min before hard_end
        conflicts = [busy(MONDAY, 9, 20)]
        slot = find_slot(p, MONDAY, conflicts)
        if slot:
            _, end = slot
            assert end.hour <= 21  # hard_end is 21:00

    def test_completed_task_still_gets_slot(self):
        # Status doesn't affect scheduling — human already confirmed it
        p = make_priority("work", status="completed")
        slot = find_slot(p, MONDAY, [])
        assert slot is not None

    def test_keyword_override_applied(self):
        # "gym" task with personal category → scheduled in health window
        p = make_priority("personal", text="gym session")
        slot = find_slot(p, MONDAY, [])
        assert slot is not None
        start, _ = slot
        assert start.hour == 6  # health window, not personal (18:00)


# ---------------------------------------------------------------------------
# schedule_day
# ---------------------------------------------------------------------------

class TestScheduleDay:
    def test_single_task_scheduled(self):
        tasks = [make_priority("work", text="deep work")]
        result = schedule_day(tasks, MONDAY, [])
        assert result[0].scheduled_start is not None
        assert result[0].scheduled_end is not None

    def test_tasks_do_not_overlap(self):
        tasks = [
            Priority(position=1, text="task A", confidence="high", category_hint="work", status="open"),
            Priority(position=2, text="task B", confidence="high", category_hint="work", status="open"),
        ]
        result = schedule_day(tasks, MONDAY, [])
        scheduled = [t for t in result if t.scheduled_start]
        if len(scheduled) == 2:
            assert scheduled[1].scheduled_start >= scheduled[0].scheduled_end

    def test_task_on_wrong_day_left_unscheduled(self):
        # work task on Sunday — no valid window
        tasks = [make_priority("work")]
        result = schedule_day(tasks, SUNDAY, [])
        assert result[0].scheduled_start is None

    def test_task_stays_on_its_day(self):
        tasks = [make_priority("work")]
        result = schedule_day(tasks, WEDNESDAY, [])
        if result[0].scheduled_start:
            assert result[0].scheduled_start.date() == WEDNESDAY

    def test_mixed_categories_scheduled_independently(self):
        tasks = [
            Priority(position=1, text="standup", confidence="high", category_hint="work", status="open"),
            Priority(position=2, text="laundry", confidence="high", category_hint="chore", status="open"),
        ]
        result = schedule_day(tasks, MONDAY, [])
        work_task = result[0]
        chore_task = result[1]
        assert work_task.scheduled_start is not None
        assert chore_task.scheduled_start is not None
        # Chore must be after 17:30
        assert chore_task.scheduled_start.hour >= 17

    def test_returns_same_number_of_tasks(self):
        tasks = [make_priority("work"), make_priority("errand")]
        result = schedule_day(tasks, MONDAY, [])
        assert len(result) == 2
