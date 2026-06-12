import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.logos.config import GRPO_OUTPUT_DIR, SFT_OUTPUT_DIR

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
CODE_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def _resolve_model_path(model_path: str | Path | None = None) -> Path:
    if model_path is None:
        if GRPO_OUTPUT_DIR.exists() and any(GRPO_OUTPUT_DIR.iterdir()):
            return GRPO_OUTPUT_DIR
        return SFT_OUTPUT_DIR

    path = Path(model_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def load_generation_pipeline(model_path: str | Path | None = None) -> tuple[Any, Any]:
    resolved_path = _resolve_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(resolved_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(resolved_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except Exception:  # noqa: BLE001
        model = AutoModelForCausalLM.from_pretrained(
            str(resolved_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return tokenizer, model


def parse_generation(text: str) -> dict[str, str]:
    think_match = THINK_PATTERN.search(text or "")
    code_match = CODE_PATTERN.search(text or "")
    return {
        "raw": text,
        "think": think_match.group(1).strip() if think_match else "",
        "answer": code_match.group(1).strip() if code_match else text.strip(),
    }


def generate_one(
    prompt: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = 768,
    temperature: float = 0.2,
) -> dict[str, str]:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text_prompt = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 1e-4),
        "top_p": 0.95,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)

    output_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    parsed = parse_generation(output_text)
    parsed["prompt"] = prompt
    return parsed
