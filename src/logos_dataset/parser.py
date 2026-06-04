import json
from typing import Iterable, List

from config import CHUNKS_DIR, RAW_PATH


def iter_raw_files() -> Iterable[str]:
    for path in sorted(CHUNKS_DIR.glob("*.txt")):
        yield str(path)


def extract_json_objects(raw_text: str) -> List[dict]:
    decoder = json.JSONDecoder()
    index = 0
    results: List[dict] = []
    length = len(raw_text)
    while index < length:
        try:
            obj, next_index = decoder.raw_decode(raw_text, idx=index)
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(obj, dict):
            results.append(obj)
        index = next_index
    return results


def _ensure_dirs() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)


def parse_raw_batches() -> int:
    _ensure_dirs()
    total = 0
    with RAW_PATH.open("w", encoding="utf-8") as handle:
        for path in iter_raw_files():
            with open(path, "r", encoding="utf-8", errors="ignore") as raw_file:
                raw_text = raw_file.read()
            objects = extract_json_objects(raw_text)
            for obj in objects:
                handle.write(json.dumps(obj, ensure_ascii=False))
                handle.write("\n")
            total += len(objects)
    return total


if __name__ == "__main__":
    total = parse_raw_batches()
    print(f"Parsed {total} objects to {RAW_PATH}")
