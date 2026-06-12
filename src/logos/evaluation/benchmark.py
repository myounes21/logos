import csv
import json
import time
from pathlib import Path
from typing import Any

from src.logos.config import RESULTS_DIR, TEST_PATH
from src.logos.core.utils import read_jsonl
from src.logos.evaluation.correctness import evaluate_correctness_rows
from src.logos.evaluation.reasoning import evaluate_reasoning_rows
from src.logos.inference.generate import generate_one, load_generation_pipeline


def _collect_latency_stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"avg_seconds": 0.0, "p50_seconds": 0.0, "p90_seconds": 0.0}
    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[int(0.5 * (len(sorted_latencies) - 1))]
    p90 = sorted_latencies[int(0.9 * (len(sorted_latencies) - 1))]
    avg = sum(sorted_latencies) / len(sorted_latencies)
    return {"avg_seconds": avg, "p50_seconds": p50, "p90_seconds": p90}


def run_benchmark(
    input_path: Path = TEST_PATH,
    model_path: Path | None = None,
    max_samples: int = 50,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> dict[str, Any]:
    rows = read_jsonl(input_path)[:max_samples]
    if not rows:
        return {
            "count": 0,
            "latency": _collect_latency_stats([]),
            "correctness": {"overall_pass_rate": 0.0},
            "reasoning": {"average_reasoning_score": 0.0},
            "outputs": [],
        }

    tokenizer, model = load_generation_pipeline(model_path=model_path)
    outputs: list[dict[str, Any]] = []
    latencies: list[float] = []

    for row in rows:
        prompt = str(row.get("instruction") or row.get("prompt") or "").strip()
        if not prompt:
            continue

        started = time.perf_counter()
        generation = generate_one(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        outputs.append(
            {
                "instruction": prompt,
                "answer": generation.get("answer", ""),
                "think": generation.get("think", ""),
                "raw": generation.get("raw", ""),
                "unit_tests": row.get("unit_tests") or [],
                "difficulty": row.get("difficulty") or "unknown",
                "topic": row.get("topic"),
                "latency_seconds": elapsed,
            }
        )

    correctness = evaluate_correctness_rows(outputs)
    reasoning = evaluate_reasoning_rows(outputs)
    return {
        "count": len(outputs),
        "latency": _collect_latency_stats(latencies),
        "correctness": {
            "overall_pass_rate": correctness.get("overall_pass_rate", 0.0),
            "by_difficulty": correctness.get("by_difficulty", {}),
        },
        "reasoning": {
            "average_reasoning_score": reasoning.get("average_reasoning_score", 0.0),
        },
        "outputs": outputs,
    }


def write_benchmark_outputs(report: dict[str, Any]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "evaluation_report.json"
    csv_path = RESULTS_DIR / "benchmark_results.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "instruction",
        "difficulty",
        "topic",
        "latency_seconds",
        "answer",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report.get("outputs", []):
            writer.writerow({field: row.get(field) for field in fields})

    return json_path, csv_path


if __name__ == "__main__":
    benchmark_report = run_benchmark()
    json_out, csv_out = write_benchmark_outputs(benchmark_report)
    print(f"Benchmark report written to {json_out}")
    print(f"Benchmark csv written to {csv_out}")
