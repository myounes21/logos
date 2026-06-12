#!/usr/bin/env bash

set -euo pipefail

python -m src.logos.cli benchmark \
  --input data/test.jsonl \
  --model-path models/grpo \
  --max-samples 50

python -m src.logos.cli eval-adversarial \
  --model-path models/grpo \
  --output results/adversarial_report.json
