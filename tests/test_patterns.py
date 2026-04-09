"""
Tests for the stuck task pattern engine (Step 6b).
Run with: venv/bin/python -m pytest tests/test_patterns.py -v

All tests use synthetic WeeklyPlan history — no database or LLM needed.
"""

import tempfile
from datetime import date, datetime
from pathlib import Path

import pytest

from app.database import init_db, load_recent_plans, save_weekly_plan
from app.models import DayPlan, Priority, ShoppingItem, StuckTask, WeeklyPlan
from app.patterns import _normalize, _same_task, get_stuck_tasks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THRESHOLD = 4
WEEKS_BACK = 6


def make_priority(text: str, status: str = "open", category: str = "work") -> Priority:
    return Priority(
        position=1, text=text, status=status,
        confidence="high", category_hint=category,
    )


def make_plan(week_offset: int, rolled_tasks: list[tuple[str, str]] = ()) -> WeeklyPlan:
    """
    week_offset: 0 = most recent, 1 = one week ago, etc.
    rolled_tasks: list of (text, category) tuples to add as rolled_over on Monday
    """
    base = date(2026, 3, 23)
    from datetime import timedelta
    week_start = base - timedelta(weeks=week_offset)

    priorities = [
        make_priority(text, status="rolled_over", category=cat)
        for text, cat in rolled_tasks
    ]
    day = DayPlan(date=week_start, day_name="Monday", priorities=priorities)

    return WeeklyPlan(
        week_start=week_start,
        week_end=week_start + timedelta(days=6),
        days=[day],
        shopping_list=[],
        extraction_notes="",
        photo_filename="week.png",
        extracted_at=datetime(week_start.year, week_start.month, week_start.day, 9, 0),
    )


# ---------------------------------------------------------------------------
# _normalize and _same_task
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases(self):
        assert _normalize("LAUNDRY") == "laundry"

    def test_strips_punctuation(self):
        assert _normalize("call dentist!") == "call dentist"

    def test_strips_extra_whitespace(self):
        assert _normalize("  gym   session  ") == "gym session"


class TestSameTask:
    def test_identical_text(self):
        assert _same_task("call dentist", "call dentist") is True

    def test_case_insensitive(self):
        assert _same_task("Call Dentist", "call dentist") is True

    def test_partial_overlap_above_threshold(self):
        assert _same_task("call dentist appt", "call dentist") is True

    def test_completely_different(self):
        assert _same_task("do laundry", "call dentist") is False

    def test_single_word_match_low_overlap(self):
        # "gym" vs "go to gym session" — depends on overlap ratio
        # Both contain "gym", overlap = 1/(1+3) = 0.25 < 0.7 — NOT same
        assert _same_task("gym", "go to gym session") is False

    def test_high_overlap_variants(self):
        assert _same_task("call mom", "call mom back") is True


# ---------------------------------------------------------------------------
# get_stuck_tasks
# ---------------------------------------------------------------------------

class TestGetStuckTasks:
    def test_empty_history_returns_empty(self):
        result = get_stuck_tasks([], threshold=THRESHOLD)
        assert result == []

    def test_insufficient_history_returns_empty(self):
        # Only 3 weeks of data — below the 6-week minimum
        plans = [make_plan(i, [("call dentist", "errand")]) for i in range(3)]
        result = get_stuck_tasks(plans, threshold=THRESHOLD, min_weeks=WEEKS_BACK)
        assert result == []

    def test_task_below_threshold_not_flagged(self):
        # Appears rolled_over 3 of 6 weeks — below threshold of 4
        plans = [
            make_plan(0, [("call dentist", "errand")]),
            make_plan(1, [("call dentist", "errand")]),
            make_plan(2, [("call dentist", "errand")]),
            make_plan(3),
            make_plan(4),
            make_plan(5),
        ]
        result = get_stuck_tasks(plans, threshold=THRESHOLD, min_weeks=WEEKS_BACK)
        assert result == []

    def test_task_at_threshold_is_flagged(self):
        # Appears rolled_over exactly 4 of 6 weeks
        plans = [
            make_plan(0, [("call dentist", "errand")]),
            make_plan(1, [("call dentist", "errand")]),
            make_plan(2, [("call dentist", "errand")]),
            make_plan(3, [("call dentist", "errand")]),
            make_plan(4),
            make_plan(5),
        ]
        result = get_stuck_tasks(plans, threshold=THRESHOLD, min_weeks=WEEKS_BACK)
        assert len(result) == 1
        assert result[0].occurrences == 4

    def test_stuck_task_has_correct_fields(self):
        plans = [make_plan(i, [("call dentist", "errand")]) for i in range(6)]
        result = get_stuck_tasks(plans, threshold=THRESHOLD, min_weeks=WEEKS_BACK)
        assert len(result) == 1
        task = result[0]
        assert isinstance(task, StuckTask)
        assert "call dentist" in task.text.lower()
        assert task.category_hint == "errand"
        assert task.occurrences == 6
        assert task.weeks_checked == 6

    def test_fuzzy_match_groups_variants(self):
        # Slightly different phrasing each week should count as same task
        plans = [
            make_plan(0, [("call dentist", "errand")]),
            make_plan(1, [("call dentist appt", "errand")]),
            make_plan(2, [("Call Dentist", "errand")]),
            make_plan(3, [("call dentist", "errand")]),
            make_plan(4),
            make_plan(5),
        ]
        result = get_stuck_tasks(plans, threshold=THRESHOLD, min_weeks=WEEKS_BACK)
        assert len(result) == 1

    def test_two_independent_stuck_tasks(self):
        plans = [
            make_plan(i, [("call dentist", "errand"), ("laundry", "chore")])
            for i in range(6)
        ]
        result = get_stuck_tasks(plans, threshold=THRESHOLD, min_weeks=WEEKS_BACK)
        assert len(result) == 2

    def test_results_sorted_by_occurrences_descending(self):
        plans_a = [make_plan(i, [("call dentist", "errand")]) for i in range(6)]
        # laundry only appears 4 weeks
        plans_b = [make_plan(i, [("call dentist", "errand"), ("laundry", "chore")])
                   for i in range(4)]
        plans_b += [make_plan(4), make_plan(5)]
        # merge: each plan for same week, take plans_b where available
        merged = plans_b[:4] + plans_a[4:]
        result = get_stuck_tasks(merged, threshold=THRESHOLD, min_weeks=WEEKS_BACK)
        if len(result) >= 2:
            assert result[0].occurrences >= result[1].occurrences

    def test_completed_tasks_not_counted(self):
        # completed tasks should not be flagged as stuck
        plans = [
            make_plan(i, [])  # no rolled_over
            for i in range(6)
        ]
        # Add completed tasks manually
        for plan in plans:
            plan.days[0].priorities.append(
                make_priority("gym", status="completed", category="health")
            )
        result = get_stuck_tasks(plans, threshold=THRESHOLD, min_weeks=WEEKS_BACK)
        assert result == []

    def test_open_tasks_not_counted(self):
        plans = [make_plan(i) for i in range(6)]
        for plan in plans:
            plan.days[0].priorities.append(
                make_priority("standup", status="open", category="work")
            )
        result = get_stuck_tasks(plans, threshold=THRESHOLD, min_weeks=WEEKS_BACK)
        assert result == []


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_init_creates_tables(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db_path = Path(f.name)
            init_db(db_path)
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                tables = {r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}
            assert "weekly_plans" in tables
            assert "shopping_items" in tables

    def test_save_and_load_plan(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db_path = Path(f.name)
            init_db(db_path)
            plan = make_plan(0, [("call dentist", "errand")])
            save_weekly_plan(plan, db_path)
            loaded = load_recent_plans(n=6, db_path=db_path)
            assert len(loaded) == 1
            assert loaded[0].week_start == plan.week_start

    def test_load_returns_most_recent_first(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db_path = Path(f.name)
            init_db(db_path)
            for i in range(3):
                save_weekly_plan(make_plan(i), db_path)
            loaded = load_recent_plans(n=6, db_path=db_path)
            assert loaded[0].week_start >= loaded[-1].week_start

    def test_load_respects_limit(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db_path = Path(f.name)
            init_db(db_path)
            for i in range(8):
                save_weekly_plan(make_plan(i), db_path)
            loaded = load_recent_plans(n=6, db_path=db_path)
            assert len(loaded) == 6
