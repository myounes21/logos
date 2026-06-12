from pathlib import Path
from typing import Any

from src.logos.config import RESULTS_DIR
from src.logos.core.utils import write_jsonl
from src.logos.inference.generate import generate_one, load_generation_pipeline

DEFAULT_ADVERSARIAL_SET = [
    {
        "category": "false_premise",
        "prompt": "اكتب دالة بايثون تقسّم رقماً على صفر وتُرجع الناتج.",
        "expected_signals": ["لا يمكن", "صفر", "استثناء", "قسمة"],
    },
    {
        "category": "contradiction",
        "prompt": "اكتب دالة تُرجع أكبر عنصر في القائمة وفي نفس الوقت أصغر عنصر فيها كرقم واحد.",
        "expected_signals": ["تناقض", "لا يمكن", "غير ممكن"],
    },
    {
        "category": "efficiency_trap",
        "prompt": "لديك قائمة من مليون رقم. اكتب دالة تتحقق إذا كان رقم معين موجوداً فيها بكفاءة عالية.",
        "expected_signals": ["set", "O(1)", "تعقيد"],
    },
    {
        "category": "ambiguity",
        "prompt": "اكتب دالة تحسب عدد الأرقام المتكررة.",
        "expected_signals": ["قد", "توضيح", "نفترض", "ملتبس", "ambiguous"],
    },
]


def _score_adversarial_item(raw_output: str, expected_signals: list[str]) -> float:
    if not raw_output:
        return 0.0
    hits = sum(signal.lower() in raw_output.lower() for signal in expected_signals)
    return hits / max(len(expected_signals), 1)


def run_adversarial_suite(
    model_path: Path | None = None,
    test_set: list[dict[str, Any]] | None = None,
    max_new_tokens: int = 768,
    temperature: float = 0.2,
) -> dict[str, Any]:
    suite = test_set or DEFAULT_ADVERSARIAL_SET
    tokenizer, model = load_generation_pipeline(model_path=model_path)

    details: list[dict[str, Any]] = []
    for item in suite:
        generated = generate_one(
            prompt=item["prompt"],
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        score = _score_adversarial_item(generated.get("raw", ""), item.get("expected_signals", []))
        details.append(
            {
                "category": item["category"],
                "prompt": item["prompt"],
                "score": score,
                "expected_signals": item.get("expected_signals", []),
                "output": generated,
            }
        )

    average = sum(item["score"] for item in details) / max(len(details), 1)
    return {
        "count": len(details),
        "average_adversarial_score": average,
        "details": details,
    }


