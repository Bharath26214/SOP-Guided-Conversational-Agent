import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _clean_env(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().strip("'").strip('"')


GROQ_API_KEY = _clean_env(os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"
LOGFIRE_TOKEN = _clean_env(os.getenv("LOGFIRE_TOKEN"))
PROMPT_GUARD_MODEL = _clean_env(os.getenv("PROMPT_GUARD_MODEL")) or "meta-llama/llama-prompt-guard-2-86m"
PROMPT_GUARD_FALLBACK_MODEL = (
    _clean_env(os.getenv("PROMPT_GUARD_FALLBACK_MODEL")) or "meta-llama/llama-prompt-guard-2-22m"
)
DUMMY_EMAIL_FROM = _clean_env(os.getenv("DUMMY_EMAIL_FROM")) or "claims-support@example.com"
BACKEND_HOST = _clean_env(os.getenv("BACKEND_HOST")) or "127.0.0.1"
BACKEND_PORT = int(_clean_env(os.getenv("BACKEND_PORT")) or "8000")
BACKEND_URL = _clean_env(os.getenv("BACKEND_URL")) or f"http://{BACKEND_HOST}:{BACKEND_PORT}"

os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")
os.environ.setdefault("LANGSMITH_TRACING", "true")

from app.observability import configure_logfire

configure_logfire()


def resolve_groq_api_key(override: str | None = None) -> str:
    return _clean_env(override) or GROQ_API_KEY
