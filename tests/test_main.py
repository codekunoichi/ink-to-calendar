"""
Tests for FastAPI routes (Step 7).
Run with: venv/bin/python -m pytest tests/test_main.py -v

All external dependencies (vision model, GCal, database) are mocked.
"""

import json
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models import (
    DayPlan, Priority, ShoppingItem, StuckTask, WeeklyPlan,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PLAN = WeeklyPlan(
    week_start=date(2026, 3, 23),
    week_end=date(2026, 3, 29),
    days=[
        DayPlan(
            date=date(2026, 3, 23),
            day_name="Monday",
            priorities=[
                Priority(
                    position=1, text="deep work", status="open",
                    confidence="high", category_hint="work",
                )
            ],
        )
    ],
    shopping_list=[
        ShoppingItem(item="almond milk", confidence="high", added_date=date(2026, 3, 23)),
        ShoppingItem(item="eggs", confidence="high", added_date=date(2026, 3, 23)),
    ],
    extraction_notes="All legible.",
    photo_filename="week.png",
    extracted_at=datetime(2026, 3, 23, 9, 0),
)

SCHEDULED_PLAN = SAMPLE_PLAN.model_copy(update={
    "days": [
        DayPlan(
            date=date(2026, 3, 23),
            day_name="Monday",
            priorities=[
                Priority(
                    position=1, text="deep work", status="open",
                    confidence="high", category_hint="work",
                    scheduled_start=datetime(2026, 3, 23, 9, 0),
                    scheduled_end=datetime(2026, 3, 23, 10, 0),
                    calendar_event_id="event_abc",
                )
            ],
        )
    ]
})


@pytest.fixture()
def client(tmp_path):
    """TestClient with database and uploads isolated to tmp_path."""
    import app.main as main_module
    with patch.object(main_module, "DB_PATH", tmp_path / "test.db"), \
         patch.object(main_module, "UPLOAD_DIR", tmp_path / "uploads"):
        from app.database import init_db
        init_db(tmp_path / "test.db")
        (tmp_path / "uploads").mkdir()
        from app.main import app
        yield TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        with patch("app.main.get_inference_client") as mock_client:
            mock_client.return_value.models.list.return_value = MagicMock()
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_status_field(self, client):
        with patch("app.main.get_inference_client") as mock_client:
            mock_client.return_value.models.list.return_value = MagicMock()
            resp = client.get("/health")
        assert "status" in resp.json()

    def test_unhealthy_when_model_unreachable(self, client):
        with patch("app.main.get_inference_client") as mock_client:
            mock_client.return_value.models.list.side_effect = Exception("connection refused")
            resp = client.get("/health")
        assert resp.json()["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

class TestUpload:
    def _upload(self, client, plan=SAMPLE_PLAN):
        with patch("app.main.extract_weekly_plan", return_value=plan):
            resp = client.post(
                "/upload",
                files={"file": ("week.png", b"fake image bytes", "image/png")},
            )
        return resp

    def test_returns_200(self, client):
        assert self._upload(client).status_code == 200

    def test_returns_weekly_plan(self, client):
        data = self._upload(client).json()
        assert "week_start" in data
        assert "days" in data

    def test_does_not_schedule(self, client):
        data = self._upload(client).json()
        for day in data["days"]:
            for task in day["priorities"]:
                assert task["scheduled_start"] is None

    def test_rejects_non_image(self, client):
        with patch("app.main.extract_weekly_plan", return_value=SAMPLE_PLAN):
            resp = client.post(
                "/upload",
                files={"file": ("notes.txt", b"some text", "text/plain")},
            )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /confirm
# ---------------------------------------------------------------------------

class TestConfirm:
    def _confirm(self, client, plan=SAMPLE_PLAN):
        with patch("app.main.schedule_week", return_value=SCHEDULED_PLAN), \
             patch("app.main.save_weekly_plan"):
            resp = client.post("/confirm", json=json.loads(plan.model_dump_json()))
        return resp

    def test_returns_200(self, client):
        assert self._confirm(client).status_code == 200

    def test_returns_summary(self, client):
        data = self._confirm(client).json()
        assert "scheduled" in data
        assert "unscheduled" in data

    def test_scheduled_count(self, client):
        data = self._confirm(client).json()
        assert data["scheduled"] == 1

    def test_persists_plan(self, client):
        with patch("app.main.schedule_week", return_value=SCHEDULED_PLAN) as _, \
             patch("app.main.save_weekly_plan") as mock_save:
            client.post("/confirm", json=json.loads(SAMPLE_PLAN.model_dump_json()))
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# GET /shopping
# ---------------------------------------------------------------------------

class TestShopping:
    def _seed(self, client):
        """Upload then confirm to seed the shopping list."""
        with patch("app.main.extract_weekly_plan", return_value=SAMPLE_PLAN):
            client.post("/upload", files={"file": ("week.png", b"img", "image/png")})
        with patch("app.main.schedule_week", return_value=SCHEDULED_PLAN), \
             patch("app.main.save_weekly_plan"):
            client.post("/confirm", json=json.loads(SAMPLE_PLAN.model_dump_json()))

    def test_returns_shopping_items(self, client):
        self._seed(client)
        resp = client.get("/shopping")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_items_default_unchecked(self, client):
        self._seed(client)
        items = client.get("/shopping").json()
        assert all(not item["checked"] for item in items)


# ---------------------------------------------------------------------------
# PATCH /shopping/{item_id}
# ---------------------------------------------------------------------------

class TestShoppingToggle:
    def _seed_and_get_id(self, client):
        with patch("app.main.extract_weekly_plan", return_value=SAMPLE_PLAN):
            client.post("/upload", files={"file": ("week.png", b"img", "image/png")})
        with patch("app.main.schedule_week", return_value=SCHEDULED_PLAN), \
             patch("app.main.save_weekly_plan"):
            client.post("/confirm", json=json.loads(SAMPLE_PLAN.model_dump_json()))
        items = client.get("/shopping").json()
        return items[0]["id"]

    def test_toggle_checked(self, client):
        item_id = self._seed_and_get_id(client)
        resp = client.patch(f"/shopping/{item_id}")
        assert resp.status_code == 200
        assert resp.json()["checked"] is True

    def test_toggle_twice_returns_unchecked(self, client):
        item_id = self._seed_and_get_id(client)
        client.patch(f"/shopping/{item_id}")
        resp = client.patch(f"/shopping/{item_id}")
        assert resp.json()["checked"] is False

    def test_invalid_id_returns_404(self, client):
        resp = client.patch("/shopping/99999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /insights
# ---------------------------------------------------------------------------

class TestInsights:
    def test_returns_list(self, client):
        with patch("app.main.load_recent_plans", return_value=[]), \
             patch("app.main.get_stuck_tasks", return_value=[]):
            resp = client.get("/insights")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_returns_stuck_tasks(self, client):
        stuck = StuckTask(
            text="call dentist", category_hint="errand",
            occurrences=4, weeks_checked=6,
        )
        with patch("app.main.load_recent_plans", return_value=[]), \
             patch("app.main.get_stuck_tasks", return_value=[stuck]):
            resp = client.get("/insights")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["text"] == "call dentist"
        assert "message" in data[0]


# ---------------------------------------------------------------------------
# GET /weeks
# ---------------------------------------------------------------------------

class TestWeeks:
    def test_returns_list(self, client):
        with patch("app.main.load_recent_plans", return_value=[SAMPLE_PLAN]):
            resp = client.get("/weeks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_returns_week_summary_fields(self, client):
        with patch("app.main.load_recent_plans", return_value=[SAMPLE_PLAN]):
            resp = client.get("/weeks")
        week = resp.json()[0]
        assert "week_start" in week
        assert "week_end" in week
        assert "task_count" in week
        assert "photo_filename" in week
