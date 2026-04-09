"""
SQLite persistence layer.

Tables:
  weekly_plans   — one row per uploaded week (full WeeklyPlan JSON)
  shopping_items — persisted shopping list items

Uses raw sqlite3 (no ORM) — the schema is simple and stable.
"""

import sqlite3
from datetime import date
from pathlib import Path

from app.models import WeeklyPlan

DEFAULT_DB = Path("ink.db")


def init_db(db_path: Path = DEFAULT_DB) -> None:
    """Create tables if they don't exist."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weekly_plans (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                week_start TEXT    NOT NULL UNIQUE,
                plan_json  TEXT    NOT NULL,
                created_at TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shopping_items (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                item       TEXT    NOT NULL,
                confidence TEXT    NOT NULL,
                checked    INTEGER DEFAULT 0,
                added_date TEXT    NOT NULL
            )
        """)


def save_weekly_plan(plan: WeeklyPlan, db_path: Path = DEFAULT_DB) -> None:
    """
    Persist a WeeklyPlan to SQLite.
    If a plan for this week_start already exists, replace it.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO weekly_plans (week_start, plan_json)
            VALUES (?, ?)
            ON CONFLICT(week_start) DO UPDATE SET plan_json = excluded.plan_json
            """,
            (str(plan.week_start), plan.model_dump_json()),
        )


def load_recent_plans(n: int = 6, db_path: Path = DEFAULT_DB) -> list[WeeklyPlan]:
    """
    Return the n most recent WeeklyPlans ordered newest-first.
    Returns fewer than n if not enough history exists.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT plan_json FROM weekly_plans ORDER BY week_start DESC LIMIT ?",
            (n,),
        ).fetchall()
    return [WeeklyPlan.model_validate_json(row[0]) for row in rows]
