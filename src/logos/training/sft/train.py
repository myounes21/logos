from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.logos.config import REPO_ROOT, SFT_OUTPUT_DIR, TRAIN_PATH, ensure_project_dirs, settings
from src.logos.core.logger import get_logger, setup_logging
from src.logos.core.utils import load_yaml
from src.logos.training.common.dataloader import LogosSFTDataset, SFTDataCollator, load_jsonl

LOGGER = get_logger(__name__)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_configs(training_config_path: Path, lora_config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    training_cfg = load_yaml(training_config_path)
    lora_cfg = load_yaml(lora_config_path)
    return training_cfg, lora_cfg


def _build_quant_config(enabled: bool, compute_dtype: str = "bfloat16") -> BitsAndBytesConfig | None:
    if not enabled:
        return None
    dtype = torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def _build_lora_config(config: dict[str, Any]) -> LoraConfig:
    target_modules = config.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return LoraConfig(
        r=int(config.get("r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        target_modules=target_modules,
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias=str(config.get("bias", "none")),
        task_type="CAUSAL_LM",
    )


def _build_training_args(config: dict[str, Any], output_dir: Path) -> TrainingArguments:
    bf16 = bool(config.get("bf16", True))
    fp16 = bool(config.get("fp16", False)) and not bf16
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(config.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        lr_scheduler_type=str(config.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(config.get("warmup_ratio", 0.05)),
        logging_steps=int(config.get("logging_steps", 10)),
        save_strategy=str(config.get("save_strategy", "epoch")),
        save_total_limit=int(config.get("save_total_limit", 2)),
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        optim=str(config.get("optim", "paged_adamw_8bit")),
        report_to=config.get("report_to", []),
        dataloader_num_workers=int(config.get("dataloader_num_workers", 0)),
        remove_unused_columns=False,
    )


def train_sft(
    data_path: Path = TRAIN_PATH,
    output_dir: Path = SFT_OUTPUT_DIR,
    training_config_path: Path = Path("configs/training_config.yaml"),
    lora_config_path: Path = Path("configs/lora_config.yaml"),
) -> Path:
    setup_logging()
    ensure_project_dirs()

    data_path = _resolve_path(data_path)
    output_dir = _resolve_path(output_dir)
    training_cfg, lora_cfg = _load_configs(
        training_config_path=_resolve_path(training_config_path),
        lora_config_path=_resolve_path(lora_config_path),
    )

    model_name = str(training_cfg.get("model_name", settings.student_model_name))
    tokenizer_name = str(training_cfg.get("tokenizer_name", settings.tokenizer_name))
    max_length = int(training_cfg.get("max_seq_length", 2048))
    use_4bit = bool(training_cfg.get("load_in_4bit", True))

    LOGGER.info("Loading dataset from %s", data_path)
    records = load_jsonl(data_path)
    if not records:
        raise ValueError(f"No records found in {data_path}")
    LOGGER.info("Loaded %d records", len(records))

    LOGGER.info("Loading tokenizer: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = _build_quant_config(
        enabled=use_4bit,
        compute_dtype=str(training_cfg.get("compute_dtype", "bfloat16")),
    )
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    LOGGER.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = _build_lora_config(lora_cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = LogosSFTDataset(records=records, tokenizer=tokenizer, max_length=max_length)
    collator = SFTDataCollator(tokenizer=tokenizer)
    args = _build_training_args(training_cfg, output_dir=output_dir)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    LOGGER.info("Starting SFT training")
    trainer.train(resume_from_checkpoint=training_cfg.get("resume_from_checkpoint", None))

    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving model artifacts to %s", output_dir)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return output_dir


if __name__ == "__main__":
    output = train_sft()
    print(f"SFT training complete. Artifacts saved to {output}")
