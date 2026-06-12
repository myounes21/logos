import re
from dataclasses import dataclass
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from src.logos.config import settings
from src.logos.core.utils import run_code_against_tests

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
CODE_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)

ARABIC_CONNECTORS = [
    "أولاً",
    "ثانياً",
    "بما أن",
    "إذن",
    "بالتالي",
    "لذلك",
    "نستنتج",
    "وعليه",
]


@dataclass
class RewardWeights:
    format_reward: float = 1.0
    correctness_reward: float = 5.0
    language_reward: float = 2.0
    logic_reward: float = 1.0
    llm_judge_reward: float = 0.0


def extract_think_and_code(text: str) -> tuple[str, str]:
    think_match = THINK_PATTERN.search(text or "")
    code_match = CODE_PATTERN.search(text or "")
    think = think_match.group(1).strip() if think_match else ""
    code = code_match.group(1).strip() if code_match else ""
    return think, code


def arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    arabic_chars = sum(1 for char in text if "\u0600" <= char <= "\u06FF")
    letters = sum(1 for char in text if char.isalpha())
    if letters == 0:
        return 0.0
    return arabic_chars / letters


def has_logic_connectors(text: str, minimum: int = 2) -> bool:
    hits = sum(connector in text for connector in ARABIC_CONNECTORS)
    return hits >= minimum

def evaluate_reasoning_trace(instruction: str, think_text: str) -> float:
    if not think_text.strip() or not settings.QWEN_API_KEY:
        return 0.0

    prompt = f"""You are an expert evaluator grading the internal "Chain-of-Thought" (reasoning trace) of an AI model trained to solve coding problems.
The model was trained to think out loud in Arabic before outputting the final Python code.

Original Problem:
{instruction}

Model's Internal Reasoning Trace (Arabic):
{think_text}

Your task is to evaluate the QUALITY of the reasoning trace above. Does the model clearly understand the problem? Does it formulate a correct logical plan?

Rate the trace on a scale of 1 to 10 based on the following criteria:
1. Understanding: Does the trace show a clear grasp of the problem requirements?
2. Logical Planning: Is the algorithmic plan sound, step-by-step, and logically correct?
3. Coherence (Arabic): Is the thought process easy to follow and clearly articulated in Arabic?

Output ONLY an integer between 1 and 10. No other text or explanation."""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.QWEN_API_KEY, base_url=settings.QWEN_BASE_URL)
        response = client.chat.completions.create(
            model=settings.QWEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a harsh but fair judge of logical reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r'\b([1-9]|10)\b', content)
        if match:
            # Normalize the 1-10 score to 0.0 - 1.0
            return float(match.group(1)) / 10.0
        return 0.1
    except Exception:
        return 0.0

def score_completion(
    completion: str,
    prompt: str = "",
    unit_tests: list[dict[str, Any]] | None = None,
    weights: RewardWeights | None = None,
) -> dict[str, Any]:
    active_weights = weights or RewardWeights()
    think, code = extract_think_and_code(completion)

    format_ok = bool(think and code and "<think>" in completion and "```" in completion)
    format_score = active_weights.format_reward if format_ok else 0.0

    language_score = 0.0
    if think and arabic_ratio(think) >= 0.6:
        language_score = active_weights.language_reward

    logic_score = active_weights.logic_reward if has_logic_connectors(think) else 0.0
    
    llm_judge_score = 0.0
    if active_weights.llm_judge_reward > 0 and think and prompt:
        normalized_judge_score = evaluate_reasoning_trace(prompt, think)
        llm_judge_score = normalized_judge_score * active_weights.llm_judge_reward

    correctness_score = 0.0
    tests = unit_tests or []
    test_result = {"passed": 0, "total": len(tests), "details": []}
    if code and tests:
        test_result = run_code_against_tests(code=code, tests=tests)
        if test_result["total"] > 0 and test_result["passed"] == test_result["total"]:
            correctness_score = active_weights.correctness_reward

    total_reward = format_score + language_score + logic_score + correctness_score + llm_judge_score
    return {
        "total": total_reward,
        "format": format_score,
        "language": language_score,
        "logic": logic_score,
        "llm_judge": llm_judge_score,
        "correctness": correctness_score,
        "tests": test_result,
        "think": think,
        "code": code,
    }


def reward_batch(
    prompts: list[str],
    completions: list[str],
    unit_tests_batch: list[list[dict[str, Any]]] | None = None,
    weights: RewardWeights | None = None,
) -> list[float]:
    tests = unit_tests_batch or [[] for _ in completions]
    prompts = prompts or ["" for _ in completions]
    
    scores: list[float] = []
    
    # We use ThreadPoolExecutor to evaluate completions in parallel, 
    # crucial for avoiding massive delays if llm_judge is enabled via network calls.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for prompt, completion, unit_tests in zip(prompts, completions, tests, strict=False):
            futures.append(
                executor.submit(
                    score_completion, 
                    completion=completion, 
                    prompt=prompt, 
                    unit_tests=unit_tests, 
                    weights=weights
                )
            )
        
        for future in futures:
            result = future.result()
            scores.append(float(result["total"]))
            
    return scores
