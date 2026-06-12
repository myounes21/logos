#!/usr/bin/env bash

set -euo pipefail

python -m src.logos.cli prepare-grpo-data \
  --input data/r1.jsonl \
  --output data/grpo.jsonl

python -m src.logos.cli train-grpo \
  --data data/grpo.jsonl \
  --output models/grpo \
  --config configs/grpo_config.yaml
