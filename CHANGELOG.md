# Changelog

## [Unreleased]

### Added

**Step 4 — Extraction test harness**
- `tests/test_extraction.py` — quality gate for planner image accuracy
  - Scans `tests/planner_images/` and runs each image through the vision pipeline
  - Scores task text accuracy (token overlap) and status mark accuracy against `_expected.json` files
  - Asserts ≥90% task accuracy; fails with prompt fix instructions if gate not met
  - CLI mode: `--generate <image>` scaffolds an expected JSON file from a new image for manual correction
  - Skips cleanly when no images are present

**Step 3 — Vision pipeline**
- `app/vision.py` — `extract_weekly_plan(image_path)` sends planner photo to Qwen2.5-VL and returns `WeeklyPlan`
  - Loads `prompts/system_prompt.txt` and `prompts/extraction_prompt.txt`
  - Strips markdown code fences from model responses before JSON parsing
  - Enriches LLM output with `photo_filename`, `extracted_at`, and shopping item `added_date`
  - Handles both `.jpg`/`.jpeg` and `.png` MIME types
- `tests/test_vision.py` — 14 unit tests (mocked) + 1 integration test (live Ollama); all passing

**Step 2 — Data models**
- `app/models.py` — Pydantic v2 models: `Priority`, `DayPlan`, `ShoppingItem`, `WeeklyPlan`, `StuckTask`
  - `Priority.status` maps handwritten planner marks: `✓` → `completed`, `>` → `rolled_over`, blank → `open`
  - `StuckTask.message` computed field generates human-readable observation for review page
- `tests/test_models.py` — 17 tests; all passing

**Step 1 — Project scaffold**
- `app/config.py` — `get_settings()` and `get_inference_client()` factory
  - Both Ollama (dev) and vLLM (prod) use the same OpenAI-compatible client; switching requires only `.env` symlink change
- `.env.example` — safe-to-commit template with placeholder values
- `.env.mac` / `.env.dgx` — environment files (gitignored); symlink active one: `ln -sf .env.mac .env`
- `docs/architecture.md` — notes on two-environment setup and backend switching
- `tests/` scaffold with `planner_images/.gitignore`

**Prompts**
- `prompts/system_prompt.txt` — added status mark detection rules (✓, >, blank)
- `prompts/extraction_prompt.txt` — added `status` field to JSON schema
- `prompts/conflict_resolution.txt` — removed stray markdown header that would have been sent to LLM

**Model update**
- Switched from `qwen2-vl:7b` (not available on Ollama) to `qwen2.5vl:7b`
- First extraction test against `test/sample_week.png`: all 7 days extracted, correct dates, shopping list separated cleanly

### Infrastructure
- Python 3.12 venv (3.14 incompatible with pydantic-core)
- `pytest.ini` with `integration` mark registered
- `.gitignore` excludes: personal planner images (`test/**/*.png/jpg`), extracted JSON, `.env.mac`, `.env.dgx`, `docs/chat_transcripts/`
