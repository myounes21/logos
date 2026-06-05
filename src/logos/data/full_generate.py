import json
import re
import time
import uuid
from pathlib import Path

from config import settings
from src.logos.core.client import client

SYSTEM_PROMPT = """
أنت مدرس برمجة خبير تشرح وتحل المسائل بأسلوب منهجي واضح.

يجب الالتزام الصارم بالتنسيق التالي:

1) فكّر بصوت عالٍ باللغة العربية الفصحى فقط.

2) التفكير يجب أن يكون:
- خطوة بخطوة
- منظم
- يشرح المنطق وليس مجرد خطوات سطحية
- يستخدم روابط منطقية مثل: "أولاً"، "بما أن"، "إذن"، "بالتالي"

3) بعد الانتهاء من التفكير:
- اكتب الكود النهائي فقط
- لا تشرح الكود بعده

4) الكود يجب أن:
- يكون بلغة Python
- صحيح وقابل للتنفيذ
- يتعامل مع الحالات الحدية (edge cases)

5) ممنوع تماماً:
- الخروج عن اللغة العربية داخل التفكير
- استخدام أي لغة أخرى

الشكل النهائي للإجابة يجب أن يكون هكذا بالضبط:

تفكير تفصيلي خطوة بخطوة...

```python
# الكود هنا
"""


def extract_code(content: str) -> str | None:
    """
    Extract python code from markdown block.
    Fallback: return raw content if no block found.
    """
    if not content:
        return None

    match = re.search(r"```python\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    generic_match = re.search(r"```\s*(.*?)```", content, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()

    return content.strip()


def is_valid_reasoning(reasoning: str) -> bool:
    """
    Basic filtering for reasoning quality.
    """
    if not reasoning:
        return False

    if len(reasoning.split()) < 30:
        return False

    bad_tokens = ["English", "中文", "英文"]
    if any(tok in reasoning for tok in bad_tokens):
        return False

    return True


def generate_sample_id(prefix: str = "logos") -> str:
    """
    Generate a stable, unique ID for each training sample.
    Format: logos_<8-char uuid hex>
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def build_sample(
    problem: str,
    reasoning: str,
    code: str,
) -> dict:
    """Return normalized training sample structure."""
    return {
        "think": reasoning.strip(),
        "answer": code.strip(),
    }


def generate_trace(problem: str) -> dict | None:
    """
    Generate one training sample from teacher model.
    Returns None if sample fails quality checks.
    """
    response = client.chat.completions.create(
        model=settings.TEACHER_LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        temperature=0.6,
        extra_body={"include_reasoning": True},
    )

    msg = response.choices[0].message
    reasoning = (getattr(msg, "reasoning", None) or "").strip()
    content = (msg.content or "").strip()
    code = extract_code(content)

    if not is_valid_reasoning(reasoning):
        return None

    if not code:
        return None

    return build_sample(problem=problem, reasoning=reasoning, code=code)


def build_output_record(record: dict, sample: dict) -> dict:
    """
    Combine the original problem record with the generated sample
    into the final schema. Single place where schema is defined —
    if you need to add a field, add it here.
    """
    return {
        "id": generate_sample_id(),
        "instruction": record.get("instruction", ""),
        "topic": record.get("topic"),
        "subtopic": record.get("subtopic"),
        "difficulty": record.get("difficulty"),
        "problem_type": record.get("problem_type"),
        "think": sample["think"],
        "answer": sample["answer"],
        "unit_tests": record.get("unit_tests"),
    }


def _load_checkpoint(checkpoint_path: Path) -> int:
    """Return the number of input lines already processed, or 0 if no checkpoint."""
    if not checkpoint_path.exists():
        return 0
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return int(data.get("lines_processed", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def _save_checkpoint(checkpoint_path: Path, lines_processed: int) -> None:
    """Persist progress so a resumed run knows where to continue from."""
    checkpoint_path.write_text(
        json.dumps({"lines_processed": lines_processed}),
        encoding="utf-8",
    )


def batch_generate(
    input_path: Path,
    output_path: Path,
    rate_limit_seconds: float = 0.5,
) -> int:
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    lines_processed = _load_checkpoint(checkpoint_path)

    if lines_processed > 0:
        print(f"Resuming from line {lines_processed}...")

    total = 0
    current_line = 0

    # Append mode so we don't overwrite already-generated samples on resume.
    file_mode = "a" if lines_processed > 0 else "w"

    with output_path.open(file_mode, encoding="utf-8") as outfile:
        with input_path.open("r", encoding="utf-8") as infile:
            for line in infile:
                current_line += 1

                # Skip lines we already handled in a previous run.
                if current_line <= lines_processed:
                    continue

                line = line.strip()
                if not line:
                    _save_checkpoint(checkpoint_path, current_line)
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    _save_checkpoint(checkpoint_path, current_line)
                    continue

                instruction = record.get("instruction", "")
                sample = generate_trace(instruction)
                if not sample:
                    _save_checkpoint(checkpoint_path, current_line)
                    continue

                output = build_output_record(record, sample)
                outfile.write(json.dumps(output, ensure_ascii=False))
                outfile.write("\n")
                outfile.flush()  # Don't lose the last record if the process dies mid-buffer.
                total += 1

                _save_checkpoint(checkpoint_path, current_line)

                if rate_limit_seconds:
                    time.sleep(rate_limit_seconds)

    # Clean up checkpoint once fully done.
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Generation complete. Checkpoint removed.")

    return total

 
if __name__ == "__main__":
    from config import TRAIN_PATH, R1_PATH

    total = batch_generate(input_path=TRAIN_PATH, output_path=R1_PATH)
    print(f"Generated {total} records to {R1_PATH}")
