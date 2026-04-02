"""
Tests for Pydantic data models (Step 2).
Run with: venv/bin/python -m pytest tests/test_models.py -v
"""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from app.models import DayPlan, Priority, ShoppingItem, StuckTask, WeeklyPlan


class TestPriority:
    def test_minimal_valid(self):
        p = Priority(position=1, text="call dentist", confidence="high", category_hint="errand")
        assert p.status == "open"
        assert p.scheduled_start is None
        assert p.scheduled_end is None
        assert p.calendar_event_id is None

    def test_status_values(self):
        for status in ("open", "completed", "rolled_over"):
            p = Priority(position=1, text="task", confidence="high", category_hint="work", status=status)
            assert p.status == status

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            Priority(position=1, text="task", confidence="high", category_hint="work", status="done")

    def test_invalid_confidence(self):
        with pytest.raises(ValidationError):
            Priority(position=1, text="task", confidence="certain", category_hint="work")

    def test_invalid_category_hint(self):
        with pytest.raises(ValidationError):
            Priority(position=1, text="task", confidence="high", category_hint="hobby")

    def test_all_category_hints(self):
        for cat in ("work", "chore", "errand", "health", "personal", "unknown"):
            p = Priority(position=1, text="task", confidence="high", category_hint=cat)
            assert p.category_hint == cat

    def test_scheduling_fields(self):
        p = Priority(
            position=1,
            text="deep work",
            confidence="high",
            category_hint="work",
            scheduled_start=datetime(2026, 3, 24, 9, 0),
            scheduled_end=datetime(2026, 3, 24, 10, 30),
            calendar_event_id="abc123",
        )
        assert p.scheduled_start.hour == 9
        assert p.calendar_event_id == "abc123"


class TestDayPlan:
    def test_valid(self):
        p = Priority(position=1, text="gym", confidence="high", category_hint="health")
        day = DayPlan(date=date(2026, 3, 24), day_name="Monday", priorities=[p])
        assert day.day_name == "Monday"
        assert len(day.priorities) == 1

    def test_empty_priorities(self):
        day = DayPlan(date=date(2026, 3, 24), day_name="Monday", priorities=[])
        assert day.priorities == []


class TestShoppingItem:
    def test_valid(self):
        item = ShoppingItem(item="almond milk", confidence="high", added_date=date(2026, 3, 24))
        assert item.checked is False

    def test_checked_toggle(self):
        item = ShoppingItem(item="eggs", confidence="high", checked=True, added_date=date(2026, 3, 24))
        assert item.checked is True

    def test_invalid_confidence(self):
        with pytest.raises(ValidationError):
            ShoppingItem(item="eggs", confidence="maybe", added_date=date(2026, 3, 24))


class TestWeeklyPlan:
    def _make_plan(self, **kwargs):
        defaults = dict(
            week_start=date(2026, 3, 23),
            week_end=date(2026, 3, 29),
            days=[],
            shopping_list=[],
            extraction_notes="",
            photo_filename="week.png",
            extracted_at=datetime(2026, 3, 23, 10, 0),
        )
        defaults.update(kwargs)
        return WeeklyPlan(**defaults)

    def test_valid(self):
        plan = self._make_plan()
        assert plan.week_start == date(2026, 3, 23)

    def test_with_days_and_shopping(self):
        priority = Priority(position=1, text="laundry", confidence="high", category_hint="chore")
        day = DayPlan(date=date(2026, 3, 24), day_name="Tuesday", priorities=[priority])
        item = ShoppingItem(item="detergent", confidence="high", added_date=date(2026, 3, 23))
        plan = self._make_plan(days=[day], shopping_list=[item])
        assert len(plan.days) == 1
        assert len(plan.shopping_list) == 1

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            WeeklyPlan(
                week_start=date(2026, 3, 23),
                week_end=date(2026, 3, 29),
                days=[],
                shopping_list=[],
                extraction_notes="",
                # photo_filename missing
                extracted_at=datetime(2026, 3, 23, 10, 0),
            )


class TestStuckTask:
    def test_valid(self):
        s = StuckTask(text="call dentist", category_hint="errand", occurrences=4, weeks_checked=6)
        assert s.occurrences == 4
        assert s.weeks_checked == 6

    def test_message(self):
        s = StuckTask(text="call dentist", category_hint="errand", occurrences=4, weeks_checked=6)
        assert "call dentist" in s.message
        assert "4" in s.message
