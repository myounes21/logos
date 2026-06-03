from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

if not ENV_PATH.exists():
    raise FileNotFoundError(f".env file not found at {ENV_PATH}")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH, extra="ignore")

    OPENROUTER_API_KEY: str
    TEACHER_LLM_MODEL: str = "deepseek/deepseek-r1"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    student_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

settings = Settings()