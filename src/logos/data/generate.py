import re

from src.logos.core.config import settings
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



def extract_code(content: str):
    """
    Extract python code from markdown block.
    Fallback: return raw content if no block found.
    """
    if not content:
        return None

    match = re.search(r"```python\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback for generic fenced blocks when language tag is missing.
    generic_match = re.search(r"```\s*(.*?)```", content, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()

    return content.strip()


def is_valid_reasoning(reasoning: str):
    """
    Basic filtering for reasoning quality.
    """
    if not reasoning:
        return False

    # too short = weak reasoning
    if len(reasoning.split()) < 30:
        return False

    # language drift (very basic)
    bad_tokens = ["English", "中文", "英文"]
    if any(tok in reasoning for tok in bad_tokens):
        return False

    return True


def build_sample(problem: str, reasoning: str, code: str):
    """Return normalized training sample structure."""
    reasoning_clean = reasoning.strip()
    code_clean = code.strip()

    full_sample = f"""<think>
{reasoning_clean}
</think>

```python
{code_clean}
```"""

    return {
        "instruction": problem,
        "think_trace": reasoning_clean,
        "answer": code_clean,
        "full_sample": full_sample,
    }



def generate_trace(problem: str):
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
