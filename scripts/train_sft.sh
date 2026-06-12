#!/usr/bin/env bash

set -euo pipefail

python -m src.logos.cli train-sft \
  --data data/r1.jsonl \
  --output models/sft \
  --training-config configs/training_config.yaml \
  --lora-config configs/lora_config.yaml
