"""
config.py
Central configuration for the Patient <-> Doctor voice translation agent.
All secrets are pulled from environment variables — never hardcode keys.
"""

import os

from dotenv import load_dotenv

# Load credentials from the project's .env file before reading configuration.
# Existing system environment variables take precedence over values in .env.
load_dotenv()

# ---------------------------------------------------------------------------
# LiveKit
# ---------------------------------------------------------------------------
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "wss://your-livekit-project.livekit.cloud")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")

# ---------------------------------------------------------------------------
# Sarvam AI (STT + TTS + Translate)
# ---------------------------------------------------------------------------
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_BASE_URL = "https://api.sarvam.ai"

# Saaras v3 handles both transcription and speech-translation via `mode`
SARVAM_STT_MODEL = "saaras:v3"
SARVAM_STT_ENDPOINT = f"{SARVAM_BASE_URL}/speech-to-text"

# Bulbul v3 for natural TTS playback back to the patient (if needed)
SARVAM_TTS_MODEL = "bulbul:v3"
SARVAM_TTS_ENDPOINT = f"{SARVAM_BASE_URL}/text-to-speech"

# Streaming (WebSocket) variants -- used by the low-latency voice pipeline so
# STT starts transcribing while audio is still arriving, and TTS starts
# playing before the whole reply has been synthesized, instead of waiting for
# one complete batch REST round-trip on each side.
SARVAM_STT_WS_ENDPOINT = "wss://api.sarvam.ai/speech-to-text/ws"
SARVAM_TTS_WS_ENDPOINT = "wss://api.sarvam.ai/text-to-speech/ws"

# Supported patient-facing languages for this deployment. Also doubles as the
# set of languages the doctor can pick as their spoken-reply target.
SUPPORTED_PATIENT_LANGUAGES = {
    "tamil": "ta-IN",
    "english": "en-IN",
    "telugu": "te-IN",
}

DOCTOR_LANGUAGE_CODE = "en-IN"

# ---------------------------------------------------------------------------
# Gemma 4 ("AI Brain") — pick ONE backend
# ---------------------------------------------------------------------------
# "google"  -> call Gemma through Google AI Studio / Gemini-compatible API (default)
# "ollama"  -> run Gemma locally, e.g. `ollama pull gemma4` then `ollama serve`
GEMMA_BACKEND = os.environ.get("GEMMA_BACKEND", "google")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
# Hosted Gemma model IDs currently supported by the Gemini API include
# "gemma-4-31b-it" and "gemma-4-26b-a4b-it". Pick the size that fits your
# latency/quality tradeoff for
# a live voice pipeline — smaller "it" (instruction-tuned) variants respond
# faster, which matters here since the doctor is waiting on each utterance.
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemma-4-26b-a4b-it")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4")

# ---------------------------------------------------------------------------
# Emergency detection keywords (fallback, deterministic safety net —
# the LLM also does semantic detection, but this catches hard trigger words
# even if the model call fails or hallucinates)
# ---------------------------------------------------------------------------
EMERGENCY_KEYWORDS_EN = [
    "chest pain", "can't breathe", "cannot breathe", "unconscious",
    "severe bleeding", "heart attack", "stroke", "seizure",
    "not breathing", "suicidal", "overdose", "collapsed",
    "breathing difficulty", "difficulty breathing", "unresponsive",
    "slurred speech", "face drooping", "one-sided weakness",
    "severe allergic reaction", "anaphylaxis", "throat swelling",
]
