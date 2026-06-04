import json
from typing import Dict, Iterable

from config import RAW_PATH, VALIDATED_PATH

REQUIRED_FIELDS = [
    "instruction",
    "topic",
    "subtopic",
    "difficulty",
    "problem_type",
    "unit_tests",
]

ALLOWED_DIFFICULTIES = {"سهل", "متوسط", "صعب"}
ALLOWED_PROBLEM_TYPES = {"كتابة دالة", "إيجاد الخطأ", "تحسين الكفاءة"}


def _ensure_dirs() -> None:
    VALIDATED_PATH.parent.mkdir(parents=True, exist_ok=True)


def _is_valid(record: Dict) -> bool:
    if not all(field in record for field in REQUIRED_FIELDS):
        return False
    if record.get("difficulty") not in ALLOWED_DIFFICULTIES:
        return False
    if record.get("problem_type") not in ALLOWED_PROBLEM_TYPES:
        return False
    unit_tests = record.get("unit_tests")
    if not isinstance(unit_tests, list) or not (3 <= len(unit_tests) <= 5):
        return False
    return True


def iter_parsed() -> Iterable[Dict]:
    with RAW_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def validate_records() -> int:
    _ensure_dirs()
    total = 0
    with VALIDATED_PATH.open("w", encoding="utf-8") as handle:
        for record in iter_parsed():
            if not _is_valid(record):
                continue
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            total += 1
    return total


if __name__ == "__main__":
    total = validate_records()
    print(f"Validated {total} records to {VALIDATED_PATH}")
