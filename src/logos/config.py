from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = REPO_ROOT / ".env"
ENV_FILE = ENV_PATH if ENV_PATH.exists() else None

DATA_DIR = REPO_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
MODELS_DIR = REPO_ROOT / "models"
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"

RAW_PATH = DATA_DIR / "raw.jsonl"
VALIDATED_PATH = DATA_DIR / "validated.jsonl"
QUALITY_PATH = DATA_DIR / "quality.jsonl"
R1_PATH = DATA_DIR / "r1.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"
GRPO_PATH = DATA_DIR / "grpo.jsonl"

SFT_OUTPUT_DIR = MODELS_DIR / "sft"
GRPO_OUTPUT_DIR = MODELS_DIR / "grpo"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    GROQ_API_KEY: str = ""
    TEACHER_LLM_MODEL: str = "deepseek-r1-distill-llama-70b"
    HF_TOKEN: str = ""

    student_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tokenizer_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    QWEN_API_KEY: str = ""
    QWEN_MODEL: str = "qwen3-coder-plus"
    QWEN_BASE_URL: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    wandb_project: str = "logos"
    wandb_run_name: str = ""


settings = Settings()


def ensure_project_dirs() -> None:
    for directory in (
        DATA_DIR,
        CHUNKS_DIR,
        MODELS_DIR,
        RUNS_DIR,
        RESULTS_DIR,
        CONFIGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
