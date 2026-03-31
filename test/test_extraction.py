"""
Extraction test harness — runs sample_week.png through the vision pipeline
and prints the raw JSON response for review.

Usage:
    venv/bin/python test/test_extraction.py

Requires Ollama running locally with qwen2-vl:7b pulled.
"""

import base64
import json
import sys
from pathlib import Path

from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5vl:7b"

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
TEST_IMAGE = Path(__file__).parent / "sample_week.png"


def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text().strip()


def encode_image(image_path: Path) -> str:
    return base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")


def run_extraction(image_path: Path) -> dict:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    system_prompt = load_prompt("system_prompt.txt")
    extraction_prompt = load_prompt("extraction_prompt.txt")
    image_b64 = encode_image(image_path)

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                    {"type": "text", "text": extraction_prompt},
                ],
            },
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if the model wraps output anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def print_summary(result: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Week: {result.get('week_start')} → {result.get('week_end')}")
    print("=" * 60)

    for day in result.get("days", []):
        priorities = day.get("priorities", [])
        if not priorities:
            continue
        print(f"\n{day['day_name']} {day['date']}")
        for p in priorities:
            status_icon = {"completed": "✓", "rolled_over": ">", "open": " "}.get(
                p.get("status", "open"), " "
            )
            confidence = p.get("confidence", "?")[0].upper()
            print(
                f"  [{status_icon}] ({confidence}) {p['text']}"
                f"  [{p.get('category_hint', 'unknown')}]"
            )

    shopping = result.get("shopping_list", [])
    if shopping:
        print("\nShopping List:")
        for item in shopping:
            print(f"  - {item['item']} ({item['confidence']})")

    notes = result.get("extraction_notes", "")
    if notes:
        print(f"\nExtraction notes: {notes}")


if __name__ == "__main__":
    if not TEST_IMAGE.exists():
        print(f"Image not found: {TEST_IMAGE}", file=sys.stderr)
        sys.exit(1)

    print(f"Running extraction on: {TEST_IMAGE.name}")
    print(f"Model: {OLLAMA_MODEL} via {OLLAMA_BASE_URL}")
    print("Sending to Ollama... (may take 30-90s on Apple Silicon)")

    try:
        result = run_extraction(TEST_IMAGE)
    except json.JSONDecodeError as e:
        print(f"\nFailed to parse JSON response: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    print_summary(result)

    # Save raw JSON for inspection
    output_path = TEST_IMAGE.parent / "sample_week_extracted.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nFull JSON saved to: {output_path}")
