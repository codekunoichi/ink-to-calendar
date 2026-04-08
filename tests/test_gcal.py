"""
Tests for Google Calendar integration and conflict resolution (Step 6).
Run with: venv/bin/python -m pytest tests/test_gcal.py -v

All tests mock the GCal API and LLM — no credentials or network needed.
"""

import json
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from app.gcal import create_calendar_event, get_busy_slots, resolve_conflict
from app.models import Priority
from app.scheduler import schedule_week
from app.models import WeeklyPlan, DayPlan, ShoppingItem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MONDAY = date(2026, 3, 23)


def make_priority(
    text: str = "task",
    category: str = "work",
    status: str = "open",
    position: int = 1,
    scheduled_start: datetime | None = None,
    scheduled_end: datetime | None = None,
) -> Priority:
    return Priority(
        position=position,
        text=text,
        confidence="high",
        category_hint=category,
        status=status,
        scheduled_start=scheduled_start,
        scheduled_end=scheduled_end,
    )


def make_mock_gcal_service(busy_periods: list[dict]) -> MagicMock:
    """Build a mock GCal service that returns given busy periods."""
    service = MagicMock()
    service.freebusy().query().execute.return_value = {
        "calendars": {
            "primary": {"busy": busy_periods}
        }
    }
    service.events().insert().execute.return_value = {"id": "event_abc123"}
    return service


def make_mock_llm_response(suggested_date: str, start: str, end: str) -> MagicMock:
    body = json.dumps({
        "suggested_date": suggested_date,
        "suggested_start": start,
        "suggested_end": end,
        "reasoning": "Best available slot given task type.",
    })
    mock_message = MagicMock()
    mock_message.content = body
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# get_busy_slots
# ---------------------------------------------------------------------------

class TestGetBusySlots:
    def test_returns_empty_when_no_busy_periods(self):
        service = make_mock_gcal_service([])
        with patch("app.gcal._build_service", return_value=service):
            slots = get_busy_slots(MONDAY)
        assert slots == []

    def test_parses_busy_periods_into_datetime_tuples(self):
        service = make_mock_gcal_service([
            {"start": "2026-03-23T09:00:00Z", "end": "2026-03-23T10:00:00Z"},
        ])
        with patch("app.gcal._build_service", return_value=service):
            slots = get_busy_slots(MONDAY)
        assert len(slots) == 1
        start, end = slots[0]
        assert isinstance(start, datetime)
        assert start.hour == 9
        assert end.hour == 10

    def test_parses_multiple_busy_periods(self):
        service = make_mock_gcal_service([
            {"start": "2026-03-23T09:00:00Z", "end": "2026-03-23T10:00:00Z"},
            {"start": "2026-03-23T14:00:00Z", "end": "2026-03-23T15:00:00Z"},
        ])
        with patch("app.gcal._build_service", return_value=service):
            slots = get_busy_slots(MONDAY)
        assert len(slots) == 2

    def test_freebusy_query_covers_full_day(self):
        service = make_mock_gcal_service([])
        with patch("app.gcal._build_service", return_value=service):
            get_busy_slots(MONDAY)
        call_body = service.freebusy().query.call_args.kwargs["body"]
        assert "2026-03-23" in call_body["timeMin"]
        assert "2026-03-23" in call_body["timeMax"] or "2026-03-24" in call_body["timeMax"]


# ---------------------------------------------------------------------------
# create_calendar_event
# ---------------------------------------------------------------------------

class TestCreateCalendarEvent:
    def _make_scheduled_task(self) -> Priority:
        return make_priority(
            text="deep work session",
            category="work",
            scheduled_start=datetime(2026, 3, 23, 9, 0),
            scheduled_end=datetime(2026, 3, 23, 10, 0),
        )

    def test_returns_event_id(self):
        service = make_mock_gcal_service([])
        task = self._make_scheduled_task()
        with patch("app.gcal._build_service", return_value=service):
            event_id = create_calendar_event(task)
        assert event_id == "event_abc123"

    def test_event_summary_matches_task_text(self):
        service = make_mock_gcal_service([])
        task = self._make_scheduled_task()
        with patch("app.gcal._build_service", return_value=service):
            create_calendar_event(task)
        event_body = service.events().insert.call_args.kwargs["body"]
        assert event_body["summary"] == "deep work session"

    def test_event_times_match_scheduled(self):
        service = make_mock_gcal_service([])
        task = self._make_scheduled_task()
        with patch("app.gcal._build_service", return_value=service):
            create_calendar_event(task)
        event_body = service.events().insert.call_args.kwargs["body"]
        assert "09:00" in event_body["start"]["dateTime"]
        assert "10:00" in event_body["end"]["dateTime"]

    def test_raises_if_task_not_scheduled(self):
        service = make_mock_gcal_service([])
        task = make_priority(text="unscheduled task")
        with patch("app.gcal._build_service", return_value=service):
            with pytest.raises(ValueError, match="not scheduled"):
                create_calendar_event(task)


# ---------------------------------------------------------------------------
# resolve_conflict
# ---------------------------------------------------------------------------

class TestResolveConflict:
    def test_returns_suggested_slot(self):
        mock_client = make_mock_llm_response("2026-03-23", "14:00", "15:00")
        open_slots = [
            (datetime(2026, 3, 23, 14, 0), datetime(2026, 3, 23, 15, 0)),
        ]
        task = make_priority(text="call dentist", category="errand")
        with patch("app.gcal.get_inference_client", return_value=mock_client):
            result = resolve_conflict(task, MONDAY, open_slots)
        assert result is not None
        start, end = result
        assert start == datetime(2026, 3, 23, 14, 0)
        assert end == datetime(2026, 3, 23, 15, 0)

    def test_returns_none_when_no_open_slots(self):
        task = make_priority(text="call dentist", category="errand")
        result = resolve_conflict(task, MONDAY, [])
        assert result is None

    def test_prompt_includes_task_text(self):
        mock_client = make_mock_llm_response("2026-03-23", "14:00", "15:00")
        open_slots = [(datetime(2026, 3, 23, 14, 0), datetime(2026, 3, 23, 15, 0))]
        task = make_priority(text="call dentist", category="errand")
        with patch("app.gcal.get_inference_client", return_value=mock_client):
            resolve_conflict(task, MONDAY, open_slots)
        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args.kwargs["messages"][0]["content"]
        assert "call dentist" in prompt_text

    def test_handles_llm_json_in_code_fence(self):
        body = json.dumps({
            "suggested_date": "2026-03-23",
            "suggested_start": "14:00",
            "suggested_end": "15:00",
            "reasoning": "Good slot.",
        })
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = f"```json\n{body}\n```"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        open_slots = [(datetime(2026, 3, 23, 14, 0), datetime(2026, 3, 23, 15, 0))]
        task = make_priority(text="call dentist", category="errand")
        with patch("app.gcal.get_inference_client", return_value=mock_client):
            result = resolve_conflict(task, MONDAY, open_slots)
        assert result is not None


# ---------------------------------------------------------------------------
# schedule_week (integration of rules engine + gcal)
# ---------------------------------------------------------------------------

class TestScheduleWeek:
    def _make_plan(self) -> WeeklyPlan:
        tasks = [
            make_priority(text="standup", category="work", position=1),
            make_priority(text="laundry", category="chore", position=2),
        ]
        day = DayPlan(date=MONDAY, day_name="Monday", priorities=tasks)
        return WeeklyPlan(
            week_start=date(2026, 3, 23),
            week_end=date(2026, 3, 29),
            days=[day],
            shopping_list=[],
            extraction_notes="",
            photo_filename="week.png",
            extracted_at=datetime(2026, 3, 23, 8, 0),
        )

    def test_scheduled_tasks_have_calendar_event_ids(self):
        service = make_mock_gcal_service([])
        plan = self._make_plan()
        with patch("app.gcal._build_service", return_value=service):
            result = schedule_week(plan)
        scheduled = [
            p for day in result.days for p in day.priorities
            if p.scheduled_start is not None
        ]
        assert all(p.calendar_event_id == "event_abc123" for p in scheduled)

    def test_returns_weekly_plan(self):
        service = make_mock_gcal_service([])
        plan = self._make_plan()
        with patch("app.gcal._build_service", return_value=service):
            result = schedule_week(plan)
        assert isinstance(result, WeeklyPlan)

    def test_unschedulable_tasks_have_no_event_id(self):
        # Block entire day so work task can't be scheduled
        service = make_mock_gcal_service([
            {"start": "2026-03-23T06:00:00Z", "end": "2026-03-23T22:00:00Z"},
        ])
        plan = self._make_plan()
        with patch("app.gcal._build_service", return_value=service), \
             patch("app.gcal.get_inference_client", return_value=MagicMock(
                 **{"chat.completions.create.return_value": MagicMock(
                     choices=[MagicMock(message=MagicMock(content="null"))]
                 )}
             )):
            result = schedule_week(plan)
        work_task = result.days[0].priorities[0]
        assert work_task.calendar_event_id is None
