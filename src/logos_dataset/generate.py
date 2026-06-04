import json
import time
from pathlib import Path
from typing import Iterable, List

from openai import OpenAI

from config import CHUNKS_DIR, RAW_PATH, settings
from logos_dataset.parser import extract_json_objects


def _ensure_dirs() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)


def iter_chunks() -> Iterable[Path]:
    return sorted(CHUNKS_DIR.glob("*.txt"))


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _qwen_client() -> OpenAI:
    return OpenAI(
        api_key=settings.QWEN_API_KEY,
        base_url=settings.QWEN_BASE_URL,
    )


def _append_raw(objects: List[dict]) -> None:
    with RAW_PATH.open("a", encoding="utf-8") as handle:
        for obj in objects:
            handle.write(json.dumps(obj, ensure_ascii=False))
            handle.write("\n")


def generate_from_chunks(rate_limit_seconds: float = 0.5) -> int:
    _ensure_dirs()
    client = _qwen_client()
    total = 0
    for chunk_path in iter_chunks():
        prompt = _load_prompt(chunk_path)
        response = client.chat.completions.create(
            model=settings.QWEN_MODEL,
            messages=[
                {"role": "system", "content": "أنت مولّد مسائل برمجية باللغة العربية."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        objects = extract_json_objects(content)
        _append_raw(objects)
        total += len(objects)
        time.sleep(rate_limit_seconds)
    return total


if __name__ == "__main__":
    total = generate_from_chunks()
    print(f"Generated {total} records to {RAW_PATH}")
