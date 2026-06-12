import re
from pathlib import Path
from statistics import mean
from typing import Any

from src.logos.config import RESULTS_DIR, TEST_PATH
from src.logos.core.utils import read_jsonl

ARABIC_CONNECTORS = [
    "أولاً",
    "ثانياً",
    "بما أن",
    "إذن",
    "بالتالي",
    "لذلك",
    "نستنتج",
    "وعليه",
]


def extract_think(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    arabic_chars = sum(1 for char in text if "\u0600" <= char <= "\u06FF")
    letters = sum(1 for char in text if char.isalpha())
    if letters == 0:
        return 0.0
    return arabic_chars / letters


def score_reasoning_text(think: str) -> dict[str, float]:
    words = think.split()
    word_count = len(words)

    length_score = min(word_count / 120.0, 1.0)
    connector_hits = sum(connector in think for connector in ARABIC_CONNECTORS)
    connector_score = min(connector_hits / 3.0, 1.0)
    language_score = _arabic_ratio(think)

    total = 0.35 * length_score + 0.30 * connector_score + 0.35 * language_score
    return {
        "total": total,
        "length_score": length_score,
        "connector_score": connector_score,
        "language_score": language_score,
        "word_count": float(word_count),
        "connector_hits": float(connector_hits),
    }


def evaluate_reasoning_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    totals: list[float] = []

    for index, row in enumerate(rows, start=1):
        think = str(row.get("think") or row.get("reasoning") or row.get("raw") or "").strip()
        if not think:
            think = extract_think(str(row.get("answer") or ""))
        score = score_reasoning_text(think)
        totals.append(score["total"])
        details.append(
            {
                "index": index,
                "instruction": row.get("instruction") or row.get("prompt"),
                "difficulty": row.get("difficulty") or "unknown",
                **score,
            }
        )

    return {
        "count": len(details),
        "average_reasoning_score": mean(totals) if totals else 0.0,
        "details": details,
    }


def evaluate_reasoning_file(input_path: Path) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    return evaluate_reasoning_rows(rows)


def write_reasoning_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    output = RESULTS_DIR / "reasoning_report.json"
    report = evaluate_reasoning_file(TEST_PATH)
    write_reasoning_report(report, output)
    print(f"Reasoning report saved to {output}")
