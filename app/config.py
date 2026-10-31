"""All settings come from .env, so no secret ever lives in code."""
import os

from dotenv import load_dotenv

load_dotenv()


def _need(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} in .env (see .env.example)")
    return value


SUPABASE_URL = _need("SUPABASE_URL")
SUPABASE_KEY = _need("SUPABASE_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "google_genai:gemini-2.5-flash")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_STT_MODE = os.getenv("SARVAM_STT_MODE", "transcribe")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "shubh")

CUSTOMER_ID = os.getenv("CUSTOMER_ID", "rahul")
DEADLINE_HOURS = 24  # simplified T+1 for the demo
