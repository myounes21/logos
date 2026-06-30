#!/bin/bash

# Ablation Study Runner
# Orchestrates the reward ablation experiments

set -e

echo "Starting Reward Ablation Study..."
echo "This will evaluate models with each reward function isolated/removed."

python -m scripts.ablate_rewards \
    --data data/grpo.jsonl \
    --output models/ablation

echo "Ablation study complete. Results saved to results/ablation_results.csv"
