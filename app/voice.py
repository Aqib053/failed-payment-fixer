"""Sarvam AI: speech to text (Saaras) and text to speech (Bulbul).
Replies are cached on disk, so repeating the demo costs no extra credits."""
import base64
import os
import hashlib
from pathlib import Path

import httpx

from app import config

BASE = "https://api.sarvam.ai"
HEADERS = {"api-subscription-key": config.SARVAM_API_KEY}
CACHE = Path("/tmp/audio_cache" if os.getenv("VERCEL") else "audio_cache")
CACHE.mkdir(exist_ok=True)


async def speech_to_text(audio: bytes) -> str:
    data = {"model": config.SARVAM_STT_MODEL, "language_code": "hi-IN"}
    if config.SARVAM_STT_MODEL.startswith("saaras"):
        data["mode"] = config.SARVAM_STT_MODE
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{BASE}/speech-to-text", headers=HEADERS, data=data,
                              files={"file": ("speech.wav", audio, "audio/wav")})
    r.raise_for_status()
    return r.json().get("transcript", "").strip()


async def text_to_speech(text: str) -> str:
    """Returns base64 WAV audio."""
    key = hashlib.sha1(f"{config.SARVAM_TTS_MODEL}|{config.SARVAM_TTS_SPEAKER}|{text}".encode()).hexdigest()
    cached = CACHE / f"{key}.wav"
    if cached.exists():
        return base64.b64encode(cached.read_bytes()).decode()
    payload = {"text": text[:1500], "target_language_code": "hi-IN",
               "model": config.SARVAM_TTS_MODEL, "speaker": config.SARVAM_TTS_SPEAKER}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{BASE}/text-to-speech", headers=HEADERS, json=payload)
    r.raise_for_status()
    audio_b64 = "".join(r.json()["audios"])
    cached.write_bytes(base64.b64decode(audio_b64))
    return audio_b64
