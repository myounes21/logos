from groq import Groq

from src.logos.config import settings


def build_groq_client() -> Groq:
    return Groq(
        api_key=settings.GROQ_API_KEY,
    )


client = build_groq_client()
