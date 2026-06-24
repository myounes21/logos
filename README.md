# LOGOS: Arabic Code-Reasoning Distillation

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**LOGOS** is an end-to-end machine learning pipeline dedicated to enhancing the Arabic code-reasoning capabilities of smaller language models. By leveraging the reasoning traces of teacher models like DeepSeek-R1, LOGOS distills complex algorithmic thinking into `Qwen/Qwen2.5-Coder-7B-Instruct` using a combination of Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).

---

## Key Features

- **Teacher Trace Distillation:** Generates high-quality algorithmic reasoning traces (`<think>` blocks) in Arabic using DeepSeek-R1 (via Groq API).
- **Efficient SFT:** Implements 4-bit QLoRA for accessible local fine-tuning.
- **Advanced GRPO Alignment:** Aligns the model using custom Arabic-aware reward functions (Correctness, Format, Language, Logic) and a dynamic **LLM-as-a-Judge** reward powered by Qwen API inside the training loop!
- **Dataset Management:** Streamlined utilities to push processed datasets directly to the Hugging Face Hub.
- **Robust Evaluation Suite:** Evaluates models on logic parsing, code correctness, and adversarial traps designed specifically for Arabic coders.
- **Modular Architecture:** Fully configurable pipelines defined through YAML configs and robust Pydantic settings.

---

## Benchmark Results

| Model | Format % | Has Function % | Code Correct % | Arabic Ratio | Logic Keywords | Tokens/sec | Reasoning Score (1-10) |
|---|---|---|---|---|---|---|---|
| **Base Qwen** | 0% | 93% | 7% | 0.00 | 0.00 | 17.30 | 1.00 |
| **SFT Only** | 80% | 95% | 18% | 0.68 | 0.80 | 15.80 | 5.50 |
| **LOGOS Full (SFT+GRPO)** | 95% | 97% | 23% | 0.82 | 1.50 | 15.70 | 7.00 |

*Note: LOGOS achieves a **3x increase** in code correctness and a **7x increase** in Arabic reasoning scores compared to the base model!*

---

## Quick Start

### 1. Environment Setup

Clone the repository and set up your Python environment using `uv` (recommended) or standard `venv`.

```bash
# Clone the repository
git clone https://github.com/yourusername/logos.git
cd logos

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Configuration

Copy the example environment file and insert your API keys:

```bash
cp .env.example .env
```
Ensure you have the following keys ready in your `.env` if you plan to use all pipeline stages:
- `GROQ_API_KEY` (for DeepSeek-R1 trace generation)
- `QWEN_API_KEY` (for Qwen LLM-as-a-judge evaluation)
- `HF_TOKEN` (for HuggingFace model access)

---

## End-to-End Workflow

The pipeline is entirely accessible via the unified CLI `src.logos.cli`. For convenience, bash scripts are also provided in the `scripts/` directory.

### Step 1: Generate Teacher Traces
Generate reasoning paths and answers using the teacher model (DeepSeek-R1).
```bash
python -m src.logos.cli generate-teacher \
  --input data/train.jsonl \
  --output data/r1.jsonl
```

### Step 2: Supervised Fine-Tuning (SFT)
Fine-tune the base `Qwen2.5-Coder` model using the teacher's traces.
```bash
python -m src.logos.cli train-sft \
  --data data/r1.jsonl \
  --output models/sft \
  --training-config configs/training_config.yaml \
  --lora-config configs/lora_config.yaml
```

### Step 3: GRPO Alignment
Align the SFT model using Group Relative Policy Optimization with custom Arabic reward functions.
```bash
# 3a. Prepare GRPO data formats
python -m src.logos.cli prepare-grpo-data \
  --input data/r1.jsonl \
  --output data/grpo.jsonl

# 3b. Train GRPO
python -m src.logos.cli train-grpo \
  --data data/grpo.jsonl \
  --output models/grpo \
  --config configs/grpo_config.yaml
```

### Step 4: Dataset Management
Push your generated training splits (`train`, `test`, `r1`) to the Hugging Face Hub for versioning and cloud access.
```bash
python scripts/push_to_hf.py
```

### Step 5: Inference & Evaluation
Test the final model's Arabic reasoning capabilities!

**Single Prompt Inference:**
```bash
python -m src.logos.cli infer \
  --model-path models/grpo \
  --prompt "اكتب دالة factorial باستخدام recursion"
```

**Full Benchmark Suite:**
```bash
python -m src.logos.cli benchmark \
  --input data/test.jsonl \
  --model-path models/grpo
```

**Evaluate Qwen Reasoning Traces:**
```bash
python scripts/eval_reasoning_qwen.py
```

---

## Project Structure

```text
logos/
├── configs/              # YAML configurations for training (SFT, LoRA, GRPO)
├── data/                 # Dataset directory (JSONL files ignored in git)
├── models/               # Output directory for model checkpoints
├── results/              # Output directory for evaluation reports
├── scripts/              # Standalone utilities and shell wrappers
├── src/logos/
│   ├── cli.py            # Unified command-line interface
│   ├── config.py         # Pydantic global settings manager
│   ├── core/             # Base utilities, clients, and loggers
│   ├── data/             # Dataset validation and filtering
│   ├── evaluation/       # Benchmark, Adversarial, and Reasoning eval suites
│   ├── inference/        # Local VLLM / Transformers generation pipeline
│   └── training/         # SFT and GRPO training loops + Reward functions
└── tests/                # Unit testing suite
```

---

## Configurations

All pipeline hyper-parameters are easily adjusted without diving into the code:
- **SFT Params:** `configs/training_config.yaml`
- **LoRA Architecture:** `configs/lora_config.yaml`
- **GRPO Rewards:** `configs/grpo_config.yaml`

---

## License
This project is open-sourced under the MIT License.
