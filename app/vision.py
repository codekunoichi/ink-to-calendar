"""
Vision pipeline — sends a planner image to Qwen2.5-VL and returns a WeeklyPlan.

The LLM returns structured JSON matching the extraction prompt schema.
This module enriches that JSON with fields not produced by the model
(photo_filename, extracted_at, shopping item dates) before validating
it into a WeeklyPlan.
"""

import base64
import json
from datetime import datetime
from pathlib import Path

from app.config import get_inference_client, get_settings
from app.models import WeeklyPlan

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text().strip()


def _encode_image(image_path: Path) -> str:
    return base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")


def _parse_response(raw: str) -> dict:
    """Parse JSON from the model response, stripping markdown code fences if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def extract_weekly_plan(image_path: Path) -> WeeklyPlan:
    """
    Send a planner image to the vision model and return a WeeklyPlan.

    The model is responsible for: task text, status marks, confidence,
    category hints, dates, and the shopping list.

    This function adds: photo_filename, extracted_at, and shopping
    item added_date (set to week_start).
    """
    settings = get_settings()
    client = get_inference_client()

    system_prompt = _load_prompt("system_prompt.txt")
    extraction_prompt = _load_prompt("extraction_prompt.txt")
    image_b64 = _encode_image(image_path)

    suffix = image_path.suffix.lower().lstrip(".")
    mime_type = "image/jpeg" if suffix in ("jpg", "jpeg") else "image/png"

    response = client.chat.completions.create(
        model=settings.inference_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                    {"type": "text", "text": extraction_prompt},
                ],
            },
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    data = _parse_response(raw)

    # Enrich with fields the LLM does not produce
    week_start = data["week_start"]
    for item in data.get("shopping_list", []):
        item.setdefault("added_date", week_start)

    data["photo_filename"] = image_path.name
    data["extracted_at"] = datetime.now().isoformat()

    return WeeklyPlan.model_validate(data)
