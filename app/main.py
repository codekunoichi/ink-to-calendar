"""
FastAPI application — ink-to-calendar (Step 7).

Routes:
    POST /upload          — extract WeeklyPlan from planner photo
    POST /confirm         — schedule confirmed plan, persist shopping list
    GET  /shopping        — list all shopping items
    PATCH /shopping/{id}  — toggle item checked/unchecked
    GET  /insights        — return stuck task observations
    GET  /weeks           — list past weekly plans (summary)
    GET  /health          — check inference backend connectivity
"""

import shutil
import sqlite3
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.auth import require_auth
from app.config import get_inference_client, get_settings
from app.database import init_db, load_recent_plans, save_weekly_plan
from app.models import WeeklyPlan
from app.patterns import get_stuck_tasks
from app.scheduler import schedule_week
from app.vision import extract_weekly_plan

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

DB_PATH = Path("ink.db")
UPLOAD_DIR = Path("uploads")
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db(DB_PATH)
    yield


app = FastAPI(
    title="ink-to-calendar",
    lifespan=lifespan,
    dependencies=[Depends(require_auth)],
)


# Static files served without auth — browser needs CSS/JS before credentials prompt
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", dependencies=[])
async def index() -> FileResponse:
    return FileResponse(str(_static_dir / "index.html"))


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """
    Accept a planner photo, run vision extraction, return WeeklyPlan for review.
    Does NOT schedule anything.
    """
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG or PNG.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    dest = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    plan = extract_weekly_plan(dest)
    return plan.model_dump(mode="json")


# ---------------------------------------------------------------------------
# POST /confirm
# ---------------------------------------------------------------------------

@app.post("/confirm")
async def confirm(plan: WeeklyPlan) -> dict:
    """
    Accept reviewed WeeklyPlan, schedule tasks, persist shopping list.
    Returns a summary of what was scheduled and what wasn't.
    """
    scheduled_plan = schedule_week(plan)
    save_weekly_plan(scheduled_plan, DB_PATH)
    _persist_shopping_list(scheduled_plan)

    scheduled = [
        p
        for day in scheduled_plan.days
        for p in day.priorities
        if p.scheduled_start is not None
    ]
    unscheduled = [
        p
        for day in scheduled_plan.days
        for p in day.priorities
        if p.scheduled_start is None
    ]

    return {
        "scheduled": len(scheduled),
        "unscheduled": len(unscheduled),
        "scheduled_tasks": [
            {
                "text": p.text,
                "day": p.scheduled_start.strftime("%A") if p.scheduled_start else None,
                "start": p.scheduled_start.strftime("%H:%M") if p.scheduled_start else None,
                "end": p.scheduled_end.strftime("%H:%M") if p.scheduled_end else None,
                "calendar_event_id": p.calendar_event_id,
            }
            for p in scheduled
        ],
        "unscheduled_tasks": [{"text": p.text, "category": p.category_hint} for p in unscheduled],
    }


def _persist_shopping_list(plan: WeeklyPlan) -> None:
    """Append new shopping items to the database (skip duplicates)."""
    with sqlite3.connect(DB_PATH) as conn:
        for item in plan.shopping_list:
            conn.execute(
                """
                INSERT OR IGNORE INTO shopping_items (item, confidence, checked, added_date)
                VALUES (?, ?, ?, ?)
                """,
                (item.item, item.confidence, int(item.checked), str(item.added_date)),
            )


# ---------------------------------------------------------------------------
# GET /shopping
# ---------------------------------------------------------------------------

@app.get("/shopping")
async def get_shopping() -> list[dict]:
    """Return all shopping items (checked and unchecked)."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, item, confidence, checked, added_date FROM shopping_items ORDER BY id"
        ).fetchall()
    return [
        {"id": r[0], "item": r[1], "confidence": r[2], "checked": bool(r[3]), "added_date": r[4]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# PATCH /shopping/{item_id}
# ---------------------------------------------------------------------------

@app.patch("/shopping/{item_id}")
async def toggle_shopping_item(item_id: int) -> dict:
    """Toggle a shopping item's checked state."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, item, confidence, checked, added_date FROM shopping_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Shopping item {item_id} not found.")
        new_checked = not bool(row[3])
        conn.execute("UPDATE shopping_items SET checked = ? WHERE id = ?", (int(new_checked), item_id))
    return {
        "id": row[0], "item": row[1], "confidence": row[2],
        "checked": new_checked, "added_date": row[4],
    }


# ---------------------------------------------------------------------------
# GET /insights
# ---------------------------------------------------------------------------

@app.get("/insights")
async def insights() -> list[dict]:
    """
    Return stuck task observations — tasks rolled_over in 4+ of last 6 weeks.
    Read-only. Called during review page load.
    """
    plans = load_recent_plans(n=6, db_path=DB_PATH)
    stuck = get_stuck_tasks(plans, threshold=4, min_weeks=6)
    return [s.model_dump() for s in stuck]


# ---------------------------------------------------------------------------
# GET /weeks
# ---------------------------------------------------------------------------

@app.get("/weeks")
async def weeks() -> list[dict]:
    """Return summary of past weekly plans (newest first)."""
    plans = load_recent_plans(n=52, db_path=DB_PATH)
    return [
        {
            "week_start": str(p.week_start),
            "week_end": str(p.week_end),
            "task_count": sum(len(d.priorities) for d in p.days),
            "photo_filename": p.photo_filename,
            "extracted_at": p.extracted_at.isoformat(),
        }
        for p in plans
    ]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Check that the inference backend is reachable."""
    settings = get_settings()
    try:
        client = get_inference_client()
        client.models.list()
        return {
            "status": "healthy",
            "backend": settings.inference_backend,
            "model": settings.inference_model,
            "url": settings.inference_base_url,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "backend": settings.inference_backend,
            "error": str(e),
        }
