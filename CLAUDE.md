# ink-to-calendar — Project Context for Claude Code

## What This Is

A personal productivity app that bridges analog planning with digital 
scheduling. The owner writes her weekly priorities in pencil in a 
physical planner every Sunday — this is intentional and therapeutic, 
not a habit to replace. This app takes a photo of that planner page 
and does four things:

1. Extracts handwritten tasks and shopping list items using a local
   vision LLM — including task status marks (✓ done, > rolled over)
2. Schedules extracted tasks into Google Calendar based on simple
   deterministic rules
3. Persists the shopping list as a mobile-accessible checklist
4. Surfaces stuck tasks — tasks that keep rolling over week after week —
   as gentle observations on the review screen

The pencil ritual stays. This app is the bridge — nothing more.

---

## Why It's Being Built This Way

- The owner follows through on tasks she writes by hand but not on 
  digital-only planning tools
- She cannot carry her planner everywhere — the digital layer solves 
  portability, not planning
- The shopping list is currently photographed or memorized — items 
  get missed
- All AI inference runs locally on a DGX Sparc. No cloud API calls 
  for the vision model. Privacy is a feature.
- The app is accessed via browser on local WiFi behind a basic login

---

## Architecture Overview
```
[Phone Camera]
     ↓ photo upload (browser)
[FastAPI backend]
     ↓ image bytes
[Qwen2-VL vision model — Ollama (dev) / vLLM (prod)]
     ↓ structured JSON (tasks + status marks + shopping list)
[Pattern engine — query SQLite history for stuck tasks]
     ↓ stuck task observations (gentle, read-only)
[Human review step — MANDATORY before any scheduling]
     ↓ confirmed tasks
[Rules-based scheduling engine]
     ↓ category → preferred time window
[Google Calendar freebusy API]
     ↓ open slot found? → create event
     ↓ no slot? → LLM fallback prompt → suggest best available slot
[Google Calendar event creation]
[Shopping list → persistent checklist UI]
```

The human review step before scheduling is non-negotiable. The app
must show the user what was extracted and let them edit/confirm before
touching the calendar.

---

## Planner Format

The physical planner is a **two-page flat spread** photographed in a
single image. One week per spread. Days are laid out in clearly bounded
sections — the vision model can use spatial position to assign tasks to
the correct day.

**Task status marks** — written by hand next to each task:

| Mark | Meaning | Extracted as |
|------|---------|--------------|
| ✓ | Completed | `completed` |
| `>` | Rolled over to another day | `rolled_over` |
| *(blank)* | Open / not yet done | `open` |

**Handwriting assumption:** The owner has clear, consistent handwriting.
The extraction prompts may assume good legibility. Do not tune prompts
for illegible handwriting — that is not the use case.

---

## Tech Stack

**Backend**
- Python 3.11+
- FastAPI (async)
- vLLM for serving Qwen2-VL locally (OpenAI-compatible API)
- Google Calendar API (via google-api-python-client)
- SQLite + SQLAlchemy for shopping list persistence
- Pydantic v2 for all data models

**Frontend**
- Vanilla HTML/CSS/JS or lightweight HTMX — no heavy framework
- Mobile-friendly (owner uploads photos from phone)
- Single-page feel: upload → review → confirm → done

**Auth**
- HTTP Basic Auth or simple token-based login
- Single user — this is a personal tool, not multi-tenant

**Infrastructure**

Two environments — code is identical, only `.env` changes:

| | MacBook Pro (dev) | DGX Sparc (production) |
|---|---|---|
| Model | Qwen2-VL 7B (4-bit) | Qwen2-VL 72B (full) |
| Serving | Ollama or MLX | vLLM |
| Endpoint | localhost:11434 | localhost:8000 |
| Access | localhost browser | local WiFi browser |
| Purpose | Prompt iteration + testing | Live weekly use |

Current phase: MacBook Pro development.
All code must work against both endpoints with only .env changes.
No cloud inference dependencies in either environment.

---

## The Four Prompts (Already Defined)

These live in `/prompts/` and must not be modified by Claude Code 
without explicit instruction.

### Prompt 1 — System Prompt (`prompts/system_prompt.txt`)
Instructs Qwen2-VL on planner structure, JSON-only output, 
confidence scoring, and extraction rules. Loaded once as the 
`system` message in every vision API call.

### Prompt 2 — Extraction Prompt (`prompts/extraction_prompt.txt`)
The `user` message sent with each planner image. Defines the exact 
JSON schema to return including week dates, daily priorities with 
confidence scores and category hints, shopping list items, and 
extraction notes.

**Expected JSON shape:**
```json
{
  "week_start": "YYYY-MM-DD",
  "week_end": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_name": "Monday",
      "priorities": [
        {
          "position": 1,
          "text": "exactly as written",
          "status": "open|completed|rolled_over",
          "confidence": "high|medium|low",
          "category_hint": "work|chore|errand|health|personal|unknown"
        }
      ]
    }
  ],
  "shopping_list": [
    {
      "item": "exactly as written",
      "confidence": "high|medium|low"
    }
  ],
  "extraction_notes": "string"
}
```

### Prompt 3 — Scheduling Rules (`prompts/rules.py`)
A Python config dictionary — NOT an LLM call. Defines:
- Category time windows (deep work before noon, chores after 5:30pm, 
  errands on Sunday mornings etc.)
- Keyword overrides (laundry → chore, gym → health, etc.)
- Default task durations per category
- Buffer minutes between tasks
- Hard start/end times for the day

Key rules the owner has specified:
- Deep work: weekday mornings before 12:00
- Chores (laundry etc.): weekdays after 17:30
- Preferred shopping day: Sunday morning 09:00–12:00
- Tasks scheduled on the day they are written for — 
  do not move a Wednesday task to Monday because Monday has open slots

### Prompt 4 — Conflict Resolution (`prompts/conflict_resolution.txt`)
Text-only LLM fallback. Only invoked when the rules engine finds 
zero open slots for a task. Returns a single suggested slot as JSON 
with one sentence of reasoning.

---

## Data Models (Pydantic)
```python
# Core extracted task
class Priority(BaseModel):
    position: int
    text: str
    status: Literal["open", "completed", "rolled_over"] = "open"  # from ✓ / > / blank
    confidence: Literal["high", "medium", "low"]
    category_hint: Literal[
        "work", "chore", "errand", "health", "personal", "unknown"
    ]
    # Set by scheduling engine, not extraction
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
```

---

## API Endpoints to Build
```
POST /upload
  - Accepts multipart image upload
  - Calls vision model with Prompts 1 + 2
  - Returns extracted WeeklyPlan JSON for review
  - Does NOT schedule anything yet

POST /confirm
  - Accepts reviewed/edited WeeklyPlan from user
  - Runs rules engine against Google Calendar freebusy
  - Creates calendar events
  - Persists shopping list to SQLite
  - Returns scheduling summary (what was scheduled, what failed)

GET /shopping
  - Returns current unchecked shopping list items

PATCH /shopping/{item_id}
  - Toggle item checked/unchecked

GET /insights
  - Called during review page load (after /upload)
  - Queries last 6 weeks of stored WeeklyPlan history
  - Returns list of stuck tasks (appeared rolled_over in 4+ of last 6 weeks)
  - Read-only — no side effects, no LLM call

GET /weeks
  - Returns list of previously uploaded weekly plans

GET /health
  - Simple health check, confirms vLLM connection is alive
```

---

## Frontend Pages to Build

### 1. Upload Page (`/`)
- Large camera/upload button (primary action, mobile-friendly)
- Instruction: "Take a photo of your Sunday planner page"
- Shows last uploaded week summary if available

### 2. Review Page (`/review`)
- Shows extracted tasks grouped by day
- Each task shows its extracted status (✓ / > / blank) — not editable,
  just displayed so she can confirm the model read the marks correctly
- Editable text fields for each task (user can correct OCR errors)
- Confidence indicators (low confidence items highlighted in amber)
- **Stuck task observation section** — appears below day tasks, above
  shopping list. Gentle plain-text list, no alarming colors. Only shown
  when there are stuck tasks to surface. Max 3 items displayed.
  Example: "call dentist has rolled over 4 of the last 6 weeks."
- Shopping list section
- Single "Schedule My Week" confirm button
- Back button to re-upload if extraction looks wrong

### 3. Shopping List Page (`/shopping`)
- Clean checklist, tap to check off items
- Persists across sessions (SQLite backed)
- Accessible from phone while in Sam's Club

### 4. Week History Page (`/weeks`)
- Simple list of past uploads with date and task count
- Tap to view a past week's plan

---

## Project Structure to Create
```
ink-to-calendar/
├── CLAUDE.md                  ← this file
├── README.md
├── .gitignore
├── requirements.txt
├── .env.example
├── prompts/
│   ├── system_prompt.txt
│   ├── extraction_prompt.txt
│   ├── conflict_resolution.txt
│   └── rules.py
├── app/
│   ├── main.py               ← FastAPI app, route registration
│   ├── config.py             ← env vars, settings
│   ├── models.py             ← all Pydantic models
│   ├── vision.py             ← inference client, extraction logic
│   │                            works with both Ollama (dev) and vLLM (prod)
│   │                            switching is handled by config.py factory
│   ├── scheduler.py          ← rules engine + GCal integration
│   ├── patterns.py           ← stuck task detection, queries WeeklyPlan history
│   ├── shopping.py           ← SQLite shopping list CRUD
│   ├── auth.py               ← basic auth middleware
│   └── static/
│       ├── index.html
│       ├── review.html
│       ├── shopping.html
│       ├── weeks.html
│       └── style.css
├── tests/
│   ├── test_extraction.py    ← runs prompts against test images, checks status mark detection
│   ├── test_scheduler.py     ← unit tests for rules engine
│   ├── test_patterns.py      ← unit tests for stuck task detection logic
│   └── planner_images/
│       └── .gitignore        ← *.jpg and *.png ignored — never commit real photos
└── docs/
    └── architecture.md
```

---

## Environment Variables

Two `.env` files — never commit either one.

**`.env.mac` (active now — MacBook Pro dev)**
```
# Ollama running Qwen2-VL 7B locally
INFERENCE_BASE_URL=http://localhost:11434/v1
INFERENCE_MODEL=qwen2-vl:7b
INFERENCE_BACKEND=ollama

# Google Calendar
GOOGLE_CREDENTIALS_PATH=./credentials.json
GOOGLE_CALENDAR_ID=primary

# App auth
APP_USERNAME=rumpa
APP_PASSWORD_HASH=<bcrypt hash>

# App settings
APP_HOST=127.0.0.1
APP_PORT=8080
DEBUG=true
```

**`.env.dgx` (production — DGX Sparc)**
```
# vLLM running Qwen2-VL 72B
INFERENCE_BASE_URL=http://localhost:8000/v1
INFERENCE_MODEL=Qwen/Qwen2-VL-72B-Instruct
INFERENCE_BACKEND=vllm

# Google Calendar
GOOGLE_CREDENTIALS_PATH=./credentials.json
GOOGLE_CALENDAR_ID=primary

# App auth
APP_USERNAME=rumpa
APP_PASSWORD_HASH=<bcrypt hash>

# App settings
APP_HOST=0.0.0.0
APP_PORT=8080
DEBUG=false
```

Symlink the active one: `ln -sf .env.mac .env`

**`.env.example`** (safe to commit — no secrets)
```
INFERENCE_BASE_URL=http://localhost:11434/v1
INFERENCE_MODEL=qwen2-vl:7b
INFERENCE_BACKEND=ollama
GOOGLE_CREDENTIALS_PATH=./credentials.json
GOOGLE_CALENDAR_ID=primary
APP_USERNAME=your_username
APP_PASSWORD_HASH=your_bcrypt_hash
APP_HOST=127.0.0.1
APP_PORT=8080
DEBUG=true
```

---

## Build Order (Start Here)

Current environment: MacBook Pro with Ollama.
Build in this sequence — each step is independently testable.

**Step 0 — Mac environment setup (do this first, once)**
```bash
# Install Ollama
brew install ollama

# Pull the dev model
ollama pull qwen2-vl:7b

# Confirm it's running
ollama serve
# → visit http://localhost:11434 to confirm

# Symlink active env
ln -sf .env.mac .env
```

**Step 1 — Project scaffold**
Folder structure, requirements.txt, config.py, .env files.
`config.py` must read `INFERENCE_BACKEND` and expose a single
`get_inference_client()` factory — Ollama and vLLM both speak
the OpenAI-compatible API, so the client code is nearly identical.

**Step 2 — Data models**
`models.py` with all Pydantic classes. No inference needed.
Fully testable on its own.

**Step 3 — Vision pipeline**
`vision.py` connecting to Ollama via Prompts 1 + 2.
Returns `WeeklyPlan`. Test with a single planner image
from the command line before wiring into FastAPI.

**Step 4 — Extraction test harness ← quality gate**
`tests/test_extraction.py` running all planner photos through the
pipeline. Track per-image accuracy — including whether ✓ and >
status marks are correctly extracted.
Do not proceed to Step 5 until accuracy is above 90%
on your own handwriting. If accuracy is low, fix the prompts
in `/prompts/` — not the code.

**Step 5 — Rules engine**
`scheduler.py` implementing the Prompt 3 config.
Unit test independently with mocked calendar data.

**Step 6 — Google Calendar integration**
freebusy queries + event creation + Prompt 4 fallback.
Test against a throwaway Google Calendar first.

**Step 6b — Pattern engine**
`patterns.py` with stuck task detection.
Input: last 6 weeks of WeeklyPlan from SQLite.
Logic: normalize task text → fuzzy match across weeks → flag tasks
that appear as `rolled_over` in 4 of last 6 weeks.
Unit test with `tests/test_patterns.py` using synthetic history.
No LLM call — pure SQL + string comparison.

**Step 7 — FastAPI routes**
`main.py` wiring all endpoints together, including `GET /insights`.

**Step 8 — Frontend**
Upload → Review → Confirm flow, then Shopping list page.
Test the full end-to-end loop on MacBook before DGX move.

**Step 9 — Auth middleware**
Basic login. Add last, after everything else works.

**Step 10 — DGX Sparc migration**
`ln -sf .env.dgx .env`
Re-run the extraction test harness against the 72B model.
Expect minor prompt tuning — budget one session for this.

Do not skip the extraction test harness in step 4. 
The owner has planner photos dating back to August that serve as 
the real-world test suite. Accuracy on those photos gates everything 
downstream.

---

## What Good Looks Like

- Upload photo on phone → confirmed calendar events in under 60 seconds
- Shopping list available on phone before leaving for Sam's Club
- Extraction accuracy above 90% on owner's handwriting
- Status marks (✓ and >) correctly extracted alongside task text
- Zero tasks scheduled on wrong days (Wednesday task stays Wednesday)
- Human always reviews before calendar is touched — no surprise events
- Stuck tasks surfaced gently after 6 weeks of history — no noise before then