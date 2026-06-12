from pathlib import Path
from statistics import mean
from typing import Any

from src.logos.config import RESULTS_DIR, TEST_PATH
from src.logos.core.utils import read_jsonl, run_code_against_tests


def _extract_code(answer: str) -> str:
    if "```" not in answer:
        return answer.strip()
    start = answer.find("```")
    end = answer.rfind("```")
    if start == end:
        return answer.replace("```", "").replace("python", "").strip()
    block = answer[start + 3 : end]
    return block.replace("python", "", 1).strip()


def evaluate_correctness_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    pass_rates: list[float] = []
    by_difficulty: dict[str, list[float]] = {}

    for idx, row in enumerate(rows, start=1):
        code = _extract_code(str(row.get("answer") or ""))
        unit_tests = row.get("unit_tests") or []
        difficulty = str(row.get("difficulty") or "unknown")
        result = run_code_against_tests(code=code, tests=unit_tests)

        total = max(int(result.get("total", 0)), 1)
        passed = int(result.get("passed", 0))
        rate = passed / total
        pass_rates.append(rate)
        by_difficulty.setdefault(difficulty, []).append(rate)

        details.append(
            {
                "index": idx,
                "instruction": row.get("instruction") or row.get("prompt"),
                "difficulty": difficulty,
                "passed": passed,
                "total": int(result.get("total", 0)),
                "pass_rate": rate,
                "test_details": result.get("details", []),
            }
        )

    aggregate = {
        "count": len(details),
        "overall_pass_rate": mean(pass_rates) if pass_rates else 0.0,
        "by_difficulty": {
            key: (mean(values) if values else 0.0)
            for key, values in sorted(by_difficulty.items())
        },
        "details": details,
    }
    return aggregate


def evaluate_correctness_file(input_path: Path) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    return evaluate_correctness_rows(rows)


def write_correctness_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    default_input = TEST_PATH
    output = RESULTS_DIR / "correctness_report.json"
    report = evaluate_correctness_file(default_input)
    write_correctness_report(report, output)
    print(f"Correctness report saved to {output}")
