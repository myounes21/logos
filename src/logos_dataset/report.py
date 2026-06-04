import json
from collections import Counter
from typing import Dict, Iterable

from config import QUALITY_PATH, RAW_PATH, VALIDATED_PATH


def _iter_jsonl(path) -> Iterable[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _count_lines(path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _print_counter(title: str, counter: Counter) -> None:
    print(title)
    for key, value in counter.most_common():
        print(f"  {key}: {value}")


def report() -> None:
    raw_count = _count_lines(RAW_PATH)
    validated_count = _count_lines(VALIDATED_PATH)
    quality_count = _count_lines(QUALITY_PATH)

    print("Counts")
    print(f"  raw: {raw_count}")
    print(f"  validated: {validated_count}")
    print(f"  quality: {quality_count}")

    topic_counter = Counter()
    difficulty_counter = Counter()
    problem_type_counter = Counter()

    for record in _iter_jsonl(QUALITY_PATH):
        topic_counter[record.get("topic", "(missing)")] += 1
        difficulty_counter[record.get("difficulty", "(missing)")] += 1
        problem_type_counter[record.get("problem_type", "(missing)")] += 1

    if quality_count:
        _print_counter("Topics", topic_counter)
        _print_counter("Difficulties", difficulty_counter)
        _print_counter("Problem Types", problem_type_counter)


if __name__ == "__main__":
    report()
