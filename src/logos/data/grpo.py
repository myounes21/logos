from pathlib import Path
from typing import Any

from src.logos.config import GRPO_PATH, R1_PATH
from src.logos.core.utils import read_jsonl, write_jsonl


def build_grpo_record(row: dict[str, Any]) -> dict[str, Any] | None:
    instruction = str(row.get("instruction") or "").strip()
    answer = str(row.get("answer") or "").strip()
    think = str(row.get("think") or "").strip()

    if not instruction or not answer:
        return None

    return {
        "prompt": instruction,
        "reference_think": think,
        "reference_answer": answer,
        "unit_tests": row.get("unit_tests") or [],
        "topic": row.get("topic"),
        "difficulty": row.get("difficulty"),
    }


def build_grpo_dataset(
    input_path: Path = R1_PATH,
    output_path: Path = GRPO_PATH,
) -> int:
    source = read_jsonl(input_path)
    rows = []
    for row in source:
        record = build_grpo_record(row)
        if record is not None:
            rows.append(record)
    return write_jsonl(output_path, rows)


if __name__ == "__main__":
    total = build_grpo_dataset()
    print(f"Prepared {total} records for GRPO at {GRPO_PATH}")
