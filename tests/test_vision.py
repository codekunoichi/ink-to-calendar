"""
Tests for the vision pipeline (Step 3).
Run with: venv/bin/python -m pytest tests/test_vision.py -v
Integration tests (require Ollama running) are marked with @pytest.mark.integration
and skipped by default. Run them with: -m integration
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.models import WeeklyPlan
from app.vision import _parse_response, extract_weekly_plan

SAMPLE_LLM_RESPONSE = {
    "week_start": "2026-03-23",
    "week_end": "2026-03-29",
    "days": [
        {
            "date": "2026-03-23",
            "day_name": "Monday",
            "priorities": [
                {
                    "position": 1,
                    "text": "call dentist",
                    "status": "rolled_over",
                    "confidence": "high",
                    "category_hint": "errand",
                }
            ],
        },
        {
            "date": "2026-03-24",
            "day_name": "Tuesday",
            "priorities": [],
        },
    ],
    "shopping_list": [
        {"item": "almond milk", "confidence": "high"},
        {"item": "eggs", "confidence": "high"},
    ],
    "extraction_notes": "All legible.",
}


def _make_mock_client(response_content: str) -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = response_content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


class TestParseResponse:
    def test_plain_json(self):
        raw = json.dumps({"key": "value"})
        result = _parse_response(raw)
        assert result == {"key": "value"}

    def test_json_in_code_fence(self):
        raw = "```\n" + json.dumps({"key": "value"}) + "\n```"
        result = _parse_response(raw)
        assert result == {"key": "value"}

    def test_json_in_typed_code_fence(self):
        raw = "```json\n" + json.dumps({"key": "value"}) + "\n```"
        result = _parse_response(raw)
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_response("not valid json")


class TestExtractWeeklyPlan:
    def _run_extract(self, response_dict: dict, image_name: str = "week.png") -> WeeklyPlan:
        mock_client = _make_mock_client(json.dumps(response_dict))
        image_path = Path(f"/tmp/{image_name}")
        image_path.write_bytes(b"fake image bytes")
        with patch("app.vision.get_inference_client", return_value=mock_client):
            return extract_weekly_plan(image_path)

    def test_returns_weekly_plan(self):
        plan = self._run_extract(SAMPLE_LLM_RESPONSE)
        assert isinstance(plan, WeeklyPlan)

    def test_week_dates_parsed(self):
        plan = self._run_extract(SAMPLE_LLM_RESPONSE)
        assert plan.week_start == date(2026, 3, 23)
        assert plan.week_end == date(2026, 3, 29)

    def test_photo_filename_set(self):
        plan = self._run_extract(SAMPLE_LLM_RESPONSE, image_name="my_week.png")
        assert plan.photo_filename == "my_week.png"

    def test_extracted_at_set(self):
        plan = self._run_extract(SAMPLE_LLM_RESPONSE)
        assert plan.extracted_at is not None

    def test_shopping_items_get_added_date(self):
        plan = self._run_extract(SAMPLE_LLM_RESPONSE)
        for item in plan.shopping_list:
            assert item.added_date == date(2026, 3, 23)

    def test_shopping_items_default_unchecked(self):
        plan = self._run_extract(SAMPLE_LLM_RESPONSE)
        for item in plan.shopping_list:
            assert item.checked is False

    def test_task_status_preserved(self):
        plan = self._run_extract(SAMPLE_LLM_RESPONSE)
        assert plan.days[0].priorities[0].status == "rolled_over"

    def test_empty_day_preserved(self):
        plan = self._run_extract(SAMPLE_LLM_RESPONSE)
        assert plan.days[1].priorities == []

    def test_jpeg_mime_type(self):
        mock_client = _make_mock_client(json.dumps(SAMPLE_LLM_RESPONSE))
        image_path = Path("/tmp/week.jpg")
        image_path.write_bytes(b"fake jpeg bytes")
        with patch("app.vision.get_inference_client", return_value=mock_client):
            plan = extract_weekly_plan(image_path)
        call_args = mock_client.chat.completions.create.call_args
        image_url = call_args.kwargs["messages"][1]["content"][0]["image_url"]["url"]
        assert image_url.startswith("data:image/jpeg")

    def test_png_mime_type(self):
        mock_client = _make_mock_client(json.dumps(SAMPLE_LLM_RESPONSE))
        image_path = Path("/tmp/week.png")
        image_path.write_bytes(b"fake png bytes")
        with patch("app.vision.get_inference_client", return_value=mock_client):
            plan = extract_weekly_plan(image_path)
        call_args = mock_client.chat.completions.create.call_args
        image_url = call_args.kwargs["messages"][1]["content"][0]["image_url"]["url"]
        assert image_url.startswith("data:image/png")


@pytest.mark.integration
class TestExtractIntegration:
    """Requires Ollama running with qwen2.5vl:7b pulled."""

    def test_sample_week(self):
        image_path = Path("test/sample_week.png")
        if not image_path.exists():
            pytest.skip("sample_week.png not found")
        plan = extract_weekly_plan(image_path)
        assert isinstance(plan, WeeklyPlan)
        assert plan.week_start is not None
        assert len(plan.days) == 7
