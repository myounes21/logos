#!/usr/bin/env bash

set -euo pipefail

python -m src.logos.cli generate-teacher \
  --input data/train.jsonl \
  --output data/r1.jsonl \
  --rate-limit 0.5
