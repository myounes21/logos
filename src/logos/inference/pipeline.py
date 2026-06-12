from pathlib import Path
from typing import Any

from src.logos.core.utils import read_jsonl, write_jsonl
from src.logos.inference.generate import generate_one, load_generation_pipeline


def run_inference_file(
    input_path: Path,
    output_path: Path,
    model_path: Path | None = None,
    max_new_tokens: int = 768,
    temperature: float = 0.2,
) -> int:
    rows = read_jsonl(input_path)
    if not rows:
        return 0

    tokenizer, model = load_generation_pipeline(model_path=model_path)
    outputs: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row.get("instruction") or row.get("prompt") or "").strip()
        if not prompt:
            continue
        generated = generate_one(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        outputs.append(
            {
                "instruction": prompt,
                "think": generated.get("think", ""),
                "answer": generated.get("answer", ""),
                "raw": generated.get("raw", ""),
                "unit_tests": row.get("unit_tests") or [],
                "topic": row.get("topic"),
                "difficulty": row.get("difficulty"),
            }
        )

    return write_jsonl(output_path, outputs)
