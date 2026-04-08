"""
Google Calendar integration and conflict resolution LLM fallback (Step 6).

get_busy_slots()       — freebusy query for a target date
create_calendar_event() — create event for a scheduled Priority
resolve_conflict()     — LLM fallback (Prompt 4) when rules engine finds no slot
"""

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.config import get_inference_client, get_settings
from app.models import Priority
from app.vision import _parse_response  # reuse JSON fence stripper

SCOPES = ["https://www.googleapis.com/auth/calendar"]
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _build_service():
    settings = get_settings()
    creds_path = Path(settings.google_credentials_path)
    token_path = creds_path.parent / "token.json"

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())

    return build("calendar", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Freebusy
# ---------------------------------------------------------------------------

def get_busy_slots(target_date: date) -> list[tuple[datetime, datetime]]:
    """
    Query Google Calendar freebusy for the given date.
    Returns list of (start, end) datetime tuples in UTC.
    """
    settings = get_settings()
    service = _build_service()

    day_start = datetime(target_date.year, target_date.month, target_date.day,
                         0, 0, 0, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    body = {
        "timeMin": day_start.isoformat(),
        "timeMax": day_end.isoformat(),
        "items": [{"id": settings.google_calendar_id}],
    }

    result = service.freebusy().query(body=body).execute()
    busy_periods = result["calendars"][settings.google_calendar_id]["busy"]

    slots = []
    for period in busy_periods:
        start = datetime.fromisoformat(period["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(period["end"].replace("Z", "+00:00"))
        # Strip timezone for comparison with naive datetimes from scheduler
        slots.append((start.replace(tzinfo=None), end.replace(tzinfo=None)))

    return slots


# ---------------------------------------------------------------------------
# Event creation
# ---------------------------------------------------------------------------

def create_calendar_event(task: Priority) -> str:
    """
    Create a Google Calendar event for a scheduled task.
    Returns the created event ID.
    Raises ValueError if task has no scheduled_start/end.
    """
    if task.scheduled_start is None or task.scheduled_end is None:
        raise ValueError(f"Task '{task.text}' is not scheduled — cannot create calendar event.")

    settings = get_settings()
    service = _build_service()

    event = {
        "summary": task.text,
        "start": {"dateTime": task.scheduled_start.isoformat()},
        "end": {"dateTime": task.scheduled_end.isoformat()},
    }

    result = service.events().insert(
        calendarId=settings.google_calendar_id,
        body=event,
    ).execute()

    return result["id"]


# ---------------------------------------------------------------------------
# Conflict resolution LLM fallback
# ---------------------------------------------------------------------------

def resolve_conflict(
    task: Priority,
    target_date: date,
    open_slots: list[tuple[datetime, datetime]],
    reason: str = "No open slot found within the category time window.",
) -> tuple[datetime, datetime] | None:
    """
    Use Prompt 4 (LLM fallback) to suggest a slot when the rules engine finds none.
    Returns (start, end) or None if open_slots is empty.
    """
    if not open_slots:
        return None

    prompt_template = (PROMPTS_DIR / "conflict_resolution.txt").read_text()

    from app.scheduler import resolve_category, _get_window  # avoid circular at module level
    category = resolve_category(task)
    window = _get_window(category, target_date)
    preferred_start = window[0].strftime("%H:%M") if window else "N/A"
    preferred_end = window[1].strftime("%H:%M") if window else "N/A"

    slots_formatted = "\n".join(
        f"- {s.strftime('%A %Y-%m-%d %H:%M')} to {e.strftime('%H:%M')}"
        for s, e in open_slots
    )

    replacements = {
        "task_text": task.text,
        "category": category,
        "preferred_start": preferred_start,
        "preferred_end": preferred_end,
        "preferred_day": target_date.strftime("%A"),
        "reason": reason,
        "open_slots_list": slots_formatted,
    }
    # Only replace {word} placeholders — leaves JSON examples like {"key": ...} untouched
    prompt = re.sub(
        r"\{(\w+)\}",
        lambda m: replacements.get(m.group(1), m.group(0)),
        prompt_template,
    )

    client = get_inference_client()
    settings = get_settings()

    response = client.chat.completions.create(
        model=settings.inference_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    try:
        data = _parse_response(raw)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    try:
        suggested_date = datetime.strptime(data["suggested_date"], "%Y-%m-%d").date()
        start = datetime.combine(
            suggested_date,
            datetime.strptime(data["suggested_start"], "%H:%M").time(),
        )
        end = datetime.combine(
            suggested_date,
            datetime.strptime(data["suggested_end"], "%H:%M").time(),
        )
        return (start, end)
    except (KeyError, ValueError):
        return None
