from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"

DATA_DIR = REPO_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"

RAW_PATH = DATA_DIR / "raw.jsonl"
VALIDATED_PATH = DATA_DIR / "validated.jsonl"
QUALITY_PATH = DATA_DIR / "quality.jsonl"
R1_PATH = DATA_DIR / "r1.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"

DEFAULT_SPEC_COUNT = 100


if not ENV_PATH.exists():
    raise FileNotFoundError(f".env file not found at {ENV_PATH}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH, extra="ignore")

    OPENROUTER_API_KEY: str
    TEACHER_LLM_MODEL: str = "deepseek/deepseek-r1"
    OPENROUTER_BASE_URL: str

    student_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    QWEN_API_KEY: str
    QWEN_MODEL: str = "qwen3-coder-plus"
    QWEN_BASE_URL: str


settings = Settings()
