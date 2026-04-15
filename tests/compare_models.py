"""
Side-by-side model comparison for planner extraction.

Runs two or more Ollama vision models against the same planner image
and reports accuracy, timing, and a diff of extracted tasks.

Usage:
    # Compare qwen2.5vl:7b vs llama3.2-vision:11b on a specific image
    venv/bin/python tests/compare_models.py tests/planner_images/sample_week.png \\
        --models qwen2.5vl:7b llama3.2-vision:11b

    # Against all images in planner_images/
    venv/bin/python tests/compare_models.py --all \\
        --models qwen2.5vl:7b llama3.2-vision:11b

    # With a custom Ollama base URL
    venv/bin/python tests/compare_models.py tests/planner_images/sample_week.png \\
        --models qwen2.5vl:7b llama3.2-vision:11b \\
        --base-url http://localhost:11434/v1

Scoring (when a _expected.json file exists alongside the image):
    - Task text accuracy: token overlap > 80% counts as a match
    - Status accuracy: extracted status == expected status
    - Shopping accuracy: token overlap > 80% on each item
    - Quality gate: >= 90% task accuracy

Output:
    A table comparing each model's scores and timing, followed by
    a per-task breakdown showing mismatches between models.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

# Resolve project root so this script works from any directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models import WeeklyPlan
from app.vision import _encode_image, _load_prompt, _parse_response

PLANNER_IMAGES_DIR = PROJECT_ROOT / "tests" / "planner_images"
ACCURACY_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Single-model extraction (bypasses config so we can inject any model name)
# ---------------------------------------------------------------------------

def extract_with_model(image_path: Path, model: str, base_url: str) -> tuple[WeeklyPlan, float]:
    """
    Run extraction with a specific model and base URL.
    Returns (WeeklyPlan, elapsed_seconds).
    """
    from datetime import datetime

    client = OpenAI(base_url=base_url, api_key="ollama")
    system_prompt = _load_prompt("system_prompt.txt")
    extraction_prompt = _load_prompt("extraction_prompt.txt")
    image_b64 = _encode_image(image_path)

    suffix = image_path.suffix.lower().lstrip(".")
    mime_type = "image/jpeg" if suffix in ("jpg", "jpeg") else "image/png"

    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
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
    elapsed = time.monotonic() - start

    raw = response.choices[0].message.content.strip()
    data = _parse_response(raw)

    week_start = data["week_start"]
    for item in data.get("shopping_list", []):
        item.setdefault("added_date", week_start)

    data["photo_filename"] = image_path.name
    data["extracted_at"] = datetime.now().isoformat()

    return WeeklyPlan.model_validate(data), elapsed


# ---------------------------------------------------------------------------
# Scoring (mirrors test_extraction.py)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(text.lower().split())


def _token_overlap(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _find_best_match(expected_text: str, extracted_tasks: list[dict]) -> dict | None:
    best, best_score = None, 0.0
    for task in extracted_tasks:
        score = _token_overlap(expected_text, task.get("text", ""))
        if score > best_score:
            best, best_score = task, score
    return best if best_score > 0.80 else None


def score_plan(expected: dict, extracted: WeeklyPlan) -> dict:
    extracted_tasks = [
        {"text": p.text, "status": p.status}
        for day in extracted.days
        for p in day.priorities
    ]

    task_results = []
    for exp_day in expected.get("days", []):
        for exp_task in exp_day.get("priorities", []):
            match = _find_best_match(exp_task["text"], extracted_tasks)
            task_results.append({
                "expected_text": exp_task["text"],
                "expected_status": exp_task.get("status", "open"),
                "matched_text": match["text"] if match else None,
                "matched_status": match["status"] if match else None,
                "text_correct": match is not None,
                "status_correct": (
                    match is not None
                    and match["status"] == exp_task.get("status", "open")
                ),
            })

    extracted_items = {item.item.lower() for item in extracted.shopping_list}
    shopping_results = []
    for exp_item in expected.get("shopping_list", []):
        item_text = exp_item["item"].lower()
        found = any(_token_overlap(item_text, ei) > 0.80 for ei in extracted_items)
        shopping_results.append({"expected": exp_item["item"], "found": found})

    total_tasks = len(task_results)
    correct_tasks = sum(1 for r in task_results if r["text_correct"])
    correct_status = sum(1 for r in task_results if r["status_correct"])
    total_shopping = len(shopping_results)
    correct_shopping = sum(1 for r in shopping_results if r["found"])

    return {
        "task_accuracy": correct_tasks / total_tasks if total_tasks else 1.0,
        "status_accuracy": correct_status / total_tasks if total_tasks else 1.0,
        "shopping_accuracy": correct_shopping / total_shopping if total_shopping else 1.0,
        "task_results": task_results,
        "shopping_results": shopping_results,
        "total_tasks": total_tasks,
        "correct_tasks": correct_tasks,
        "correct_status": correct_status,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _col(width: int, text: str) -> str:
    return str(text)[:width].ljust(width)


def print_comparison(image_path: Path, results: dict[str, tuple[WeeklyPlan, float, dict | None]]) -> None:
    models = list(results.keys())
    sep = "─" * 72

    print(f"\n{'═' * 72}")
    print(f"  Image: {image_path.name}")
    print(f"{'═' * 72}")

    # Summary table
    header = f"  {'Model':<32}  {'Tasks':>6}  {'Text%':>6}  {'Status%':>8}  {'Shop%':>6}  {'Time':>6}"
    print(header)
    print(f"  {sep}")

    for model, (plan, elapsed, scores) in results.items():
        total_extracted = sum(len(d.priorities) for d in plan.days)
        if scores:
            row = (
                f"  {_col(32, model)}"
                f"  {total_extracted:>6}"
                f"  {scores['task_accuracy']:>5.0%}"
                f"  {scores['status_accuracy']:>7.0%}"
                f"  {scores['shopping_accuracy']:>5.0%}"
                f"  {elapsed:>5.1f}s"
            )
            gate = "✓" if scores["task_accuracy"] >= ACCURACY_THRESHOLD else "✗"
            row += f"  {gate}"
        else:
            row = (
                f"  {_col(32, model)}"
                f"  {total_extracted:>6}"
                f"  {'n/a':>5}"
                f"  {'n/a':>7}"
                f"  {'n/a':>5}"
                f"  {elapsed:>5.1f}s"
            )
        print(row)

    # Per-task breakdown when expected file is present
    first_scores = next(
        (s for _, _, s in results.values() if s is not None), None
    )
    if first_scores is None:
        print(f"\n  (No expected file found — showing raw extraction per model)\n")
        for model, (plan, _, _) in results.items():
            print(f"\n  ── {model} ──")
            for day in plan.days:
                if day.priorities:
                    print(f"    {day.day_name} {day.date}")
                    for p in day.priorities:
                        icon = {"completed": "✓", "rolled_over": ">", "open": " "}.get(p.status, " ")
                        print(f"      [{icon}] {p.text}  [{p.category_hint}]")
            if plan.shopping_list:
                print(f"    Shopping:")
                for item in plan.shopping_list:
                    print(f"      - {item.item}")
        return

    # Diff: show tasks where models disagree or where any model missed
    print(f"\n  Task-level breakdown (expected vs each model):")
    print(f"  {sep}")

    task_results_by_model = {
        model: scores["task_results"]
        for model, (_, _, scores) in results.items()
        if scores
    }

    if not task_results_by_model:
        return

    first_model = list(task_results_by_model.keys())[0]
    all_tasks = task_results_by_model[first_model]

    disagreements = 0
    for i, task in enumerate(all_tasks):
        per_model = {
            m: task_results_by_model[m][i]
            for m in task_results_by_model
            if i < len(task_results_by_model[m])
        }
        # Show if any model missed or if status differs
        any_miss = any(not r["text_correct"] for r in per_model.values())
        any_status_wrong = any(
            r["text_correct"] and not r["status_correct"] for r in per_model.values()
        )
        models_disagree = len({r["text_correct"] for r in per_model.values()}) > 1

        if any_miss or any_status_wrong or models_disagree:
            disagreements += 1
            print(f"\n  Expected: \"{task['expected_text']}\"  (status: {task['expected_status']})")
            for model, r in per_model.items():
                if not r["text_correct"]:
                    status = "MISSED"
                elif not r["status_correct"]:
                    status = f"wrong status → got '{r['matched_status']}'"
                else:
                    status = "OK"
                print(f"    {_col(32, model)}  {status}")

    if disagreements == 0:
        print("\n  All models agree and match expected output.")

    # Shopping diffs
    shopping_by_model = {
        model: scores["shopping_results"]
        for model, (_, _, scores) in results.items()
        if scores
    }
    if shopping_by_model:
        first_shopping = list(shopping_by_model.values())[0]
        shop_disagree = []
        for i, item in enumerate(first_shopping):
            per_model = {m: shopping_by_model[m][i] for m in shopping_by_model if i < len(shopping_by_model[m])}
            if any(not r["found"] for r in per_model.values()):
                shop_disagree.append((item["expected"], per_model))
        if shop_disagree:
            print(f"\n  Shopping mismatches:")
            for expected_item, per_model in shop_disagree:
                print(f"  Expected: \"{expected_item}\"")
                for model, r in per_model.items():
                    print(f"    {_col(32, model)}  {'found' if r['found'] else 'MISSED'}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(image_path: Path, models: list[str], base_url: str) -> None:
    expected_path = image_path.parent / f"{image_path.stem}_expected.json"
    expected = json.loads(expected_path.read_text()) if expected_path.exists() else None

    results: dict[str, tuple[WeeklyPlan, float, dict | None]] = {}

    for model in models:
        print(f"  Running {model} on {image_path.name}...", end=" ", flush=True)
        try:
            plan, elapsed = extract_with_model(image_path, model, base_url)
            scores = score_plan(expected, plan) if expected else None
            results[model] = (plan, elapsed, scores)
            print(f"done ({elapsed:.1f}s)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            results[model] = (None, 0.0, None)

    # Remove failed extractions before printing
    results = {m: v for m, v in results.items() if v[0] is not None}
    if results:
        print_comparison(image_path, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two or more Ollama vision models on planner extraction.")
    parser.add_argument("image", nargs="?", help="Path to a single planner image")
    parser.add_argument("--all", action="store_true", help="Run on all images in tests/planner_images/")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5vl:7b", "llama3.2-vision:11b"],
        metavar="MODEL",
        help="Ollama model names to compare (default: qwen2.5vl:7b llama3.2-vision:11b)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="Ollama API base URL (default: http://localhost:11434/v1)",
    )
    args = parser.parse_args()

    if args.all:
        images = sorted(
            p for p in PLANNER_IMAGES_DIR.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            and not p.stem.endswith("_expected")
        )
        if not images:
            print(f"No images found in {PLANNER_IMAGES_DIR}")
            sys.exit(1)
    elif args.image:
        images = [Path(args.image)]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\nModels under comparison:")
    for m in args.models:
        print(f"  - {m}")

    for image_path in images:
        run_comparison(image_path, args.models, args.base_url)


if __name__ == "__main__":
    main()
