"""
Extraction test harness — Step 4 quality gate.

Runs all planner images in tests/planner_images/ through the vision pipeline
and scores accuracy against expected output files (if present).

Usage:
    # Run harness (requires Ollama):
    venv/bin/python -m pytest tests/test_extraction.py -v -m integration -s

    # Create expected output for a new image (run once, then edit to correct):
    venv/bin/python tests/test_extraction.py --generate tests/planner_images/week_aug_04.jpg

Accuracy scoring:
    - A task is "correct" if its text has >80% token overlap with an expected task
    - Status mark is "correct" if it matches the expected status exactly
    - Overall accuracy = correct tasks / total expected tasks
    - Quality gate: >= 90% accuracy required to proceed to Step 5

Expected file format: {image_stem}_expected.json alongside the image.
Example: tests/planner_images/week_aug_04_expected.json
"""

import json
import sys
from pathlib import Path

import pytest

from app.models import WeeklyPlan
from app.vision import extract_weekly_plan

PLANNER_IMAGES_DIR = Path(__file__).parent / "planner_images"
ACCURACY_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(text.lower().split())


def _token_overlap(a: str, b: str) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _find_best_match(expected_text: str, extracted_tasks: list[dict]) -> dict | None:
    """Return the extracted task with the highest token overlap to expected_text."""
    best, best_score = None, 0.0
    for task in extracted_tasks:
        score = _token_overlap(expected_text, task.get("text", ""))
        if score > best_score:
            best, best_score = task, score
    return best if best_score > 0.80 else None


def score_plan(expected: dict, extracted: WeeklyPlan) -> dict:
    """
    Compare extracted WeeklyPlan against expected JSON.
    Returns a score dict with task accuracy, status accuracy, and shopping accuracy.
    """
    extracted_tasks = [
        p.__dict__
        for day in extracted.days
        for p in day.priorities
    ]

    task_results = []
    for exp_day in expected.get("days", []):
        for exp_task in exp_day.get("priorities", []):
            match = _find_best_match(exp_task["text"], extracted_tasks)
            task_results.append({
                "expected": exp_task["text"],
                "matched": match["text"] if match else None,
                "text_correct": match is not None,
                "status_correct": (
                    match is not None
                    and match.get("status", "open") == exp_task.get("status", "open")
                ),
            })

    # Shopping list scoring
    extracted_items = {item.item.lower() for item in extracted.shopping_list}
    shopping_results = []
    for exp_item in expected.get("shopping_list", []):
        item_text = exp_item["item"].lower()
        tokens = _tokenize(item_text)
        found = any(
            _token_overlap(item_text, ei) > 0.80 for ei in extracted_items
        )
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
    }


def print_report(image_path: Path, extracted: WeeklyPlan, scores: dict | None) -> None:
    print(f"\n{'=' * 60}")
    print(f"Image: {image_path.name}")
    print(f"Week:  {extracted.week_start} → {extracted.week_end}")
    print(f"Days with tasks: {sum(1 for d in extracted.days if d.priorities)}/7")
    print(f"Total tasks extracted: {sum(len(d.priorities) for d in extracted.days)}")
    print(f"Shopping items: {len(extracted.shopping_list)}")

    if scores:
        print(f"\nAccuracy vs expected:")
        print(f"  Task text:    {scores['task_accuracy']:.0%}  ({scores['correct_tasks']}/{scores['total_tasks']})")
        print(f"  Status marks: {scores['status_accuracy']:.0%}")
        print(f"  Shopping:     {scores['shopping_accuracy']:.0%}")

        failures = [r for r in scores["task_results"] if not r["text_correct"]]
        if failures:
            print(f"\nMissed tasks ({len(failures)}):")
            for f in failures:
                print(f"  - expected: {f['expected']}")

        status_failures = [r for r in scores["task_results"] if r["text_correct"] and not r["status_correct"]]
        if status_failures:
            print(f"\nStatus mismatches ({len(status_failures)}):")
            for f in status_failures:
                print(f"  - '{f['matched']}': wrong status")

        gate = scores["task_accuracy"] >= ACCURACY_THRESHOLD
        print(f"\nQuality gate (≥{ACCURACY_THRESHOLD:.0%}): {'✓ PASS' if gate else '✗ FAIL'}")
    else:
        print("\n(No expected file — manual review only)")
        for day in extracted.days:
            if day.priorities:
                print(f"\n  {day.day_name} {day.date}")
                for p in day.priorities:
                    icon = {"completed": "✓", "rolled_over": ">", "open": " "}.get(p.status, " ")
                    print(f"    [{icon}] {p.text}  [{p.category_hint}]")
        if extracted.shopping_list:
            print("\n  Shopping:")
            for item in extracted.shopping_list:
                print(f"    - {item.item}")


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

def _collect_images() -> list[Path]:
    if not PLANNER_IMAGES_DIR.exists():
        return []
    return sorted(
        p for p in PLANNER_IMAGES_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        and not p.stem.endswith("_expected")
    )


@pytest.mark.integration
@pytest.mark.parametrize("image_path", _collect_images(), ids=lambda p: p.stem)
def test_extraction_accuracy(image_path: Path) -> None:
    """Runs extraction and asserts accuracy >= 90% when expected file exists."""
    extracted = extract_weekly_plan(image_path)
    assert isinstance(extracted, WeeklyPlan), "Extraction did not return a WeeklyPlan"

    expected_path = image_path.parent / f"{image_path.stem}_expected.json"
    scores = None
    if expected_path.exists():
        expected = json.loads(expected_path.read_text())
        scores = score_plan(expected, extracted)

    print_report(image_path, extracted, scores)

    if scores is not None:
        assert scores["task_accuracy"] >= ACCURACY_THRESHOLD, (
            f"Task accuracy {scores['task_accuracy']:.0%} is below the {ACCURACY_THRESHOLD:.0%} gate. "
            "Fix the prompts in /prompts/ before proceeding to Step 5."
        )


@pytest.mark.integration
def test_no_images_is_not_a_failure() -> None:
    """Passes when planner_images/ is empty — harness is ready, images not yet added."""
    images = _collect_images()
    if not images:
        pytest.skip("No planner images found in tests/planner_images/ — add images to run harness")


# ---------------------------------------------------------------------------
# CLI: generate expected output scaffold
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate expected output scaffold for an image.")
    parser.add_argument("--generate", metavar="IMAGE_PATH", help="Path to planner image")
    args = parser.parse_args()

    if args.generate:
        image_path = Path(args.generate)
        print(f"Extracting from {image_path.name}...")
        plan = extract_weekly_plan(image_path)
        expected_path = image_path.parent / f"{image_path.stem}_expected.json"

        scaffold = {
            "week_start": str(plan.week_start),
            "week_end": str(plan.week_end),
            "days": [
                {
                    "date": str(day.date),
                    "day_name": day.day_name,
                    "priorities": [
                        {
                            "position": p.position,
                            "text": p.text,
                            "status": p.status,
                            "confidence": p.confidence,
                            "category_hint": p.category_hint,
                        }
                        for p in day.priorities
                    ],
                }
                for day in plan.days
                if day.priorities
            ],
            "shopping_list": [
                {"item": item.item, "confidence": item.confidence}
                for item in plan.shopping_list
            ],
        }

        expected_path.write_text(json.dumps(scaffold, indent=2))
        print(f"Scaffold saved to {expected_path}")
        print("Review and correct it, then re-run the harness to score accuracy.")
