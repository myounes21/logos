import json
import random
from typing import Dict, List

from src.logos.config import QUALITY_PATH, TEST_PATH, TRAIN_PATH


def _ensure_dirs() -> None:
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_quality() -> List[Dict]:
    records: List[Dict] = []
    with QUALITY_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def split_dataset(test_ratio: float = 0.1, seed: int = 42) -> Dict[str, int]:
    _ensure_dirs()
    records = _load_quality()
    random.Random(seed).shuffle(records)

    test_size = int(len(records) * test_ratio)
    test_records = records[:test_size]
    train_records = records[test_size:]

    with TRAIN_PATH.open("w", encoding="utf-8") as train_handle:
        for record in train_records:
            train_handle.write(json.dumps(record, ensure_ascii=False))
            train_handle.write("\n")

    with TEST_PATH.open("w", encoding="utf-8") as test_handle:
        for record in test_records:
            test_handle.write(json.dumps(record, ensure_ascii=False))
            test_handle.write("\n")

    return {"train": len(train_records), "test": len(test_records)}


if __name__ == "__main__":
    counts = split_dataset()
    print(f"Split into train={counts['train']} test={counts['test']}")
