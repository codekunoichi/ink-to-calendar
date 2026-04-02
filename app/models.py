"""
Pydantic v2 data models for ink-to-calendar.

Priority.status is extracted directly from handwritten marks in the planner:
  ✓  → completed
  >  → rolled_over
  (blank) → open
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, computed_field


class Priority(BaseModel):
    position: int
    text: str
    status: Literal["open", "completed", "rolled_over"] = "open"
    confidence: Literal["high", "medium", "low"]
    category_hint: Literal["work", "chore", "errand", "health", "personal", "unknown"]
    # Set by scheduling engine after human review — not populated by extraction
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    calendar_event_id: str | None = None


class DayPlan(BaseModel):
    date: date
    day_name: str
    priorities: list[Priority]


class ShoppingItem(BaseModel):
    item: str
    confidence: Literal["high", "medium", "low"]
    checked: bool = False
    added_date: date


class WeeklyPlan(BaseModel):
    week_start: date
    week_end: date
    days: list[DayPlan]
    shopping_list: list[ShoppingItem]
    extraction_notes: str
    photo_filename: str
    extracted_at: datetime


class StuckTask(BaseModel):
    """A task that has appeared as rolled_over in 4+ of the last 6 weeks."""

    text: str
    category_hint: Literal["work", "chore", "errand", "health", "personal", "unknown"]
    occurrences: int
    weeks_checked: int

    @computed_field
    @property
    def message(self) -> str:
        return f'"{self.text}" has rolled over {self.occurrences} of the last {self.weeks_checked} weeks.'
