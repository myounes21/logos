import json
import re
from typing import Dict, Iterable

from config import QUALITY_PATH, VALIDATED_PATH


def _ensure_dirs() -> None:
    QUALITY_PATH.parent.mkdir(parents=True, exist_ok=True)


def iter_validated() -> Iterable[Dict]:
    with VALIDATED_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _has_duplicate_tests(unit_tests: list) -> bool:
    seen = set()
    for test in unit_tests:
        normalized = json.dumps(test, sort_keys=True, ensure_ascii=False)
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


def _looks_like_linked_list(text: str) -> bool:
    return "قائمة مرتبطة" in text or "linked list" in text.lower()


def _tests_look_like_list(unit_tests: list) -> bool:
    for test in unit_tests:
        payload = json.dumps(test, ensure_ascii=False)
        if "[" in payload and "]" in payload:
            return True
    return False


def _is_ambiguous_instruction(instruction: str) -> bool:
    ambiguous_terms = [
        "minimum operations",
        "أقل عدد من العمليات",
        "الحد الأدنى من العمليات",
    ]
    if any(term in instruction.lower() for term in ambiguous_terms):
        if "عملية" in instruction and "تعريف" not in instruction:
            return True
    return False


def passes_quality(record: Dict) -> bool:
    instruction = (record.get("instruction") or "").strip()
    unit_tests = record.get("unit_tests") or []

    if _has_duplicate_tests(unit_tests):
        return False

    if _looks_like_linked_list(instruction) and _tests_look_like_list(unit_tests):
        return False

    if _is_ambiguous_instruction(instruction):
        return False

    return True


def filter_quality() -> int:
    _ensure_dirs()
    total = 0
    with QUALITY_PATH.open("w", encoding="utf-8") as handle:
        for record in iter_validated():
            if not passes_quality(record):
                continue
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            total += 1
    return total


if __name__ == "__main__":
    total = filter_quality()
    print(f"Quality filtered {total} records to {QUALITY_PATH}")
