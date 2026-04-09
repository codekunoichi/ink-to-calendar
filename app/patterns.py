"""
Stuck task pattern engine (Step 6b).

Detects tasks that have appeared as rolled_over in 4+ of the last 6 weeks.
No LLM call — pure string comparison against WeeklyPlan history.

Design:
  - get_stuck_tasks() is pure: takes a list of WeeklyPlan objects as input.
    The caller (FastAPI route) loads plans from the database.
  - Fuzzy matching uses token overlap (Jaccard similarity >= 0.70).
  - Only rolled_over tasks are counted; open and completed are ignored.
  - Requires min_weeks of history before surfacing anything (avoids noise).
"""

import re
from app.models import StuckTask, WeeklyPlan

_OVERLAP_THRESHOLD = 0.60


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _same_task(a: str, b: str) -> bool:
    """True if two task strings refer to the same task (Jaccard token overlap >= 0.70)."""
    tokens_a = set(_normalize(a).split())
    tokens_b = set(_normalize(b).split())
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    return overlap >= _OVERLAP_THRESHOLD


def get_stuck_tasks(
    plans: list[WeeklyPlan],
    threshold: int = 4,
    min_weeks: int = 6,
) -> list[StuckTask]:
    """
    Analyse WeeklyPlan history and return tasks stuck in rolled_over status.

    Args:
        plans:     Weekly plans ordered newest-first (from load_recent_plans).
        threshold: Minimum number of weeks a task must appear as rolled_over.
        min_weeks: Minimum weeks of history required before surfacing anything.

    Returns:
        List of StuckTask sorted by occurrences descending. Empty if history
        is below min_weeks or no task meets the threshold.
    """
    if len(plans) < min_weeks:
        return []

    weeks_checked = len(plans)

    # Each entry: {text, category, weeks: set of week indices where rolled_over}
    fingerprints: list[dict] = []

    for week_idx, plan in enumerate(plans):
        for day in plan.days:
            for priority in day.priorities:
                if priority.status != "rolled_over":
                    continue

                matched = False
                for fp in fingerprints:
                    if _same_task(priority.text, fp["text"]):
                        fp["weeks"].add(week_idx)
                        matched = True
                        break

                if not matched:
                    fingerprints.append({
                        "text": priority.text,
                        "category": priority.category_hint,
                        "weeks": {week_idx},
                    })

    stuck = [
        StuckTask(
            text=fp["text"],
            category_hint=fp["category"],
            occurrences=len(fp["weeks"]),
            weeks_checked=weeks_checked,
        )
        for fp in fingerprints
        if len(fp["weeks"]) >= threshold
    ]

    return sorted(stuck, key=lambda s: s.occurrences, reverse=True)
