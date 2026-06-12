import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def normalize_sample(record: dict[str, Any]) -> dict[str, Any] | None:
    instruction = str(record.get("instruction") or "").strip()
    think = str(record.get("think") or record.get("think_trace") or "").strip()
    answer = str(record.get("answer") or "").strip()

    if not instruction or not answer:
        return None

    normalized = {
        "instruction": instruction,
        "think": think,
        "answer": _strip_code_fences(answer),
        "topic": record.get("topic"),
        "difficulty": record.get("difficulty"),
        "unit_tests": record.get("unit_tests") or [],
    }
    return normalized


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        sample = normalize_sample(record)
        if sample is not None:
            normalized.append(sample)
    return normalized


def render_assistant_content(think: str, answer: str) -> str:
    answer = _strip_code_fences(answer)
    if think:
        return f"<think>\n{think}\n</think>\n\n```python\n{answer}\n```"
    return f"```python\n{answer}\n```"


def build_chat_text(tokenizer: Any, instruction: str, think: str, answer: str) -> tuple[str, str]:
    user_message = {"role": "user", "content": instruction}
    assistant_message = {
        "role": "assistant",
        "content": render_assistant_content(think=think, answer=answer),
    }

    if hasattr(tokenizer, "apply_chat_template"):
        full_text = tokenizer.apply_chat_template(
            [user_message, assistant_message],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = f"User: {instruction}\nAssistant:"
        full_text = f"{prompt_text}\n{assistant_message['content']}"

    return full_text, prompt_text


class LogosSFTDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.records = normalize_records(records)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.records[index]
        full_text, prompt_text = build_chat_text(
            tokenizer=self.tokenizer,
            instruction=row["instruction"],
            think=row["think"],
            answer=row["answer"],
        )

        encoded_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded_prompt = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded_full["input_ids"][0]
        attention_mask = encoded_full["attention_mask"][0]
        labels = input_ids.clone()

        prompt_length = int(encoded_prompt["input_ids"].shape[-1])
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class SFTDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_padded = torch.full_like(batch["input_ids"], fill_value=-100)
        for i, label in enumerate(labels):
            labels_padded[i, : label.shape[0]] = label

        batch["labels"] = labels_padded
        return batch
