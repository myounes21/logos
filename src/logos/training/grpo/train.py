from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.logos.config import GRPO_OUTPUT_DIR, GRPO_PATH, REPO_ROOT, ensure_project_dirs, settings
from src.logos.core.logger import get_logger, setup_logging
from src.logos.core.utils import load_yaml, read_jsonl
from src.logos.training.grpo.rewards import RewardWeights, reward_batch

LOGGER = get_logger(__name__)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _build_reward_weights(config: dict[str, Any]) -> RewardWeights:
    return RewardWeights(
        format_reward=float(config.get("format_reward", 1.0)),
        correctness_reward=float(config.get("correctness_reward", 5.0)),
        language_reward=float(config.get("language_reward", 2.0)),
        logic_reward=float(config.get("logic_reward", 1.0)),
        llm_judge_reward=float(config.get("llm_judge_reward", 0.0)),
    )


def _to_hf_dataset(rows: list[dict[str, Any]]) -> Dataset:
    samples = []
    for row in rows:
        prompt = str(row.get("prompt") or row.get("instruction") or "").strip()
        if not prompt:
            continue
        samples.append(
            {
                "prompt": prompt,
                "unit_tests": row.get("unit_tests") or [],
                "difficulty": row.get("difficulty"),
                "topic": row.get("topic"),
            }
        )
    if not samples:
        raise ValueError("No valid GRPO rows found")
    return Dataset.from_list(samples)


def train_grpo(
    data_path: Path = GRPO_PATH,
    output_dir: Path = GRPO_OUTPUT_DIR,
    config_path: Path = Path("configs/grpo_config.yaml"),
) -> Path:
    setup_logging()
    ensure_project_dirs()

    data_path = _resolve_path(data_path)
    output_dir = _resolve_path(output_dir)
    config = load_yaml(_resolve_path(config_path))

    rows = read_jsonl(data_path)
    dataset = _to_hf_dataset(rows)
    model_name_or_path = str(config.get("model_name_or_path", settings.student_model_name))
    tokenizer_name = str(config.get("tokenizer_name", model_name_or_path))

    weights = _build_reward_weights(config)

    def reward_function(prompts: list[str], completions: list[str], **kwargs: Any) -> list[float]:
        unit_tests = kwargs.get("unit_tests") or [[] for _ in completions]
        return reward_batch(prompts=prompts, completions=completions, unit_tests_batch=unit_tests, weights=weights)

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "trl is required for GRPO training. Install dependencies from pyproject.toml first."
        ) from exc

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=float(config.get("learning_rate", 1e-6)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
        logging_steps=int(config.get("logging_steps", 10)),
        num_generations=int(config.get("num_generations", 8)),
        max_prompt_length=int(config.get("max_prompt_length", 1024)),
        max_completion_length=int(config.get("max_completion_length", 1024)),
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        bf16=bool(config.get("bf16", True)),
        fp16=bool(config.get("fp16", False)),
        report_to=config.get("report_to", []),
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    LOGGER.info("Starting GRPO training with %d rows", len(dataset))
    trainer.train()
    trainer.save_model(str(output_dir))

    return output_dir


if __name__ == "__main__":
    output = train_grpo()
    print(f"GRPO training complete. Artifacts saved to {output}")
