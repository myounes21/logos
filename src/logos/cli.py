import argparse
import json
from pathlib import Path

from src.logos.config import (
    GRPO_PATH,
    R1_PATH,
    RESULTS_DIR,
    SFT_OUTPUT_DIR,
    TEST_PATH,
    TRAIN_PATH,
)
def _add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=Path, default=None)


def cmd_generate_teacher(args: argparse.Namespace) -> None:
    from src.logos.data.full_generate import batch_generate

    total = batch_generate(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        rate_limit_seconds=args.rate_limit,
    )
    print(f"Generated {total} records to {args.output}")


def cmd_prepare_grpo(args: argparse.Namespace) -> None:
    from src.logos.data.grpo import build_grpo_dataset

    total = build_grpo_dataset(input_path=args.input, output_path=args.output)
    print(f"Prepared {total} GRPO records at {args.output}")


def cmd_train_sft(args: argparse.Namespace) -> None:
    from src.logos.training.sft.train import train_sft

    output = train_sft(
        data_path=args.data,
        output_dir=args.output,
        training_config_path=args.training_config,
        lora_config_path=args.lora_config,
    )
    print(f"SFT artifacts saved at {output}")


def cmd_train_grpo(args: argparse.Namespace) -> None:
    from src.logos.training.grpo.train import train_grpo

    output = train_grpo(
        data_path=args.data,
        output_dir=args.output,
        config_path=args.config,
    )
    print(f"GRPO artifacts saved at {output}")


def cmd_infer(args: argparse.Namespace) -> None:
    from src.logos.inference.generate import generate_one, load_generation_pipeline
    from src.logos.inference.pipeline import run_inference_file

    if args.prompt:
        tokenizer, model = load_generation_pipeline(model_path=args.model_path)
        generated = generate_one(
            prompt=args.prompt,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(json.dumps(generated, ensure_ascii=False, indent=2))
        return

    if not args.input or not args.output:
        raise ValueError("For batch inference you must provide both --input and --output")

    total = run_inference_file(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"Generated predictions for {total} rows to {args.output}")


def cmd_eval_correctness(args: argparse.Namespace) -> None:
    from src.logos.evaluation.correctness import evaluate_correctness_file, write_correctness_report

    report = evaluate_correctness_file(args.input)
    output = args.output or (RESULTS_DIR / "correctness_report.json")
    write_correctness_report(report, output)
    print(f"Correctness report saved to {output}")


def cmd_eval_reasoning(args: argparse.Namespace) -> None:
    from src.logos.evaluation.reasoning import evaluate_reasoning_file, write_reasoning_report

    report = evaluate_reasoning_file(args.input)
    output = args.output or (RESULTS_DIR / "reasoning_report.json")
    write_reasoning_report(report, output)
    print(f"Reasoning report saved to {output}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from src.logos.evaluation.benchmark import run_benchmark, write_benchmark_outputs

    report = run_benchmark(
        input_path=args.input,
        model_path=args.model_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    json_path, csv_path = write_benchmark_outputs(report)
    print(f"Benchmark json: {json_path}")
    print(f"Benchmark csv: {csv_path}")


def cmd_adversarial(args: argparse.Namespace) -> None:
    from src.logos.evaluation.adversarial import run_adversarial_suite

    report = run_adversarial_suite(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    output = args.output or (RESULTS_DIR / "adversarial_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Adversarial report saved to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LOGOS pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_teacher = subparsers.add_parser("generate-teacher", help="Generate think+answer traces")
    generate_teacher.add_argument("--input", type=Path, default=TRAIN_PATH)
    generate_teacher.add_argument("--output", type=Path, default=R1_PATH)
    generate_teacher.add_argument("--checkpoint", type=Path, default=None)
    generate_teacher.add_argument("--rate-limit", type=float, default=0.5)
    generate_teacher.set_defaults(func=cmd_generate_teacher)

    prepare_grpo = subparsers.add_parser("prepare-grpo-data", help="Build GRPO dataset from R1 output")
    prepare_grpo.add_argument("--input", type=Path, default=R1_PATH)
    prepare_grpo.add_argument("--output", type=Path, default=GRPO_PATH)
    prepare_grpo.set_defaults(func=cmd_prepare_grpo)

    train_sft_parser = subparsers.add_parser("train-sft", help="Run QLoRA SFT training")
    train_sft_parser.add_argument("--data", type=Path, default=TRAIN_PATH)
    train_sft_parser.add_argument("--output", type=Path, default=SFT_OUTPUT_DIR)
    train_sft_parser.add_argument("--training-config", type=Path, default=Path("configs/training_config.yaml"))
    train_sft_parser.add_argument("--lora-config", type=Path, default=Path("configs/lora_config.yaml"))
    train_sft_parser.set_defaults(func=cmd_train_sft)

    train_grpo_parser = subparsers.add_parser("train-grpo", help="Run GRPO training")
    train_grpo_parser.add_argument("--data", type=Path, default=GRPO_PATH)
    train_grpo_parser.add_argument("--output", type=Path, default=Path("models/grpo"))
    train_grpo_parser.add_argument("--config", type=Path, default=Path("configs/grpo_config.yaml"))
    train_grpo_parser.set_defaults(func=cmd_train_grpo)

    infer_parser = subparsers.add_parser("infer", help="Run generation")
    infer_parser.add_argument("--prompt", type=str, default="")
    infer_parser.add_argument("--input", type=Path, default=None)
    infer_parser.add_argument("--output", type=Path, default=None)
    infer_parser.add_argument("--max-new-tokens", type=int, default=768)
    infer_parser.add_argument("--temperature", type=float, default=0.2)
    _add_shared_model_args(infer_parser)
    infer_parser.set_defaults(func=cmd_infer)

    eval_correctness = subparsers.add_parser("eval-correctness", help="Evaluate unit-test correctness")
    eval_correctness.add_argument("--input", type=Path, default=TEST_PATH)
    eval_correctness.add_argument("--output", type=Path, default=None)
    eval_correctness.set_defaults(func=cmd_eval_correctness)

    eval_reasoning = subparsers.add_parser("eval-reasoning", help="Evaluate reasoning quality")
    eval_reasoning.add_argument("--input", type=Path, default=TEST_PATH)
    eval_reasoning.add_argument("--output", type=Path, default=None)
    eval_reasoning.set_defaults(func=cmd_eval_reasoning)

    benchmark = subparsers.add_parser("benchmark", help="Run end-to-end benchmark")
    benchmark.add_argument("--input", type=Path, default=TEST_PATH)
    benchmark.add_argument("--max-samples", type=int, default=50)
    benchmark.add_argument("--max-new-tokens", type=int, default=512)
    benchmark.add_argument("--temperature", type=float, default=0.2)
    _add_shared_model_args(benchmark)
    benchmark.set_defaults(func=cmd_benchmark)

    adversarial = subparsers.add_parser("eval-adversarial", help="Run adversarial stress tests")
    adversarial.add_argument("--output", type=Path, default=None)
    adversarial.add_argument("--max-new-tokens", type=int, default=768)
    adversarial.add_argument("--temperature", type=float, default=0.2)
    _add_shared_model_args(adversarial)
    adversarial.set_defaults(func=cmd_adversarial)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
