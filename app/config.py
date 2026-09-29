from dotenv import load_dotenv
import os

load_dotenv()

# Configuration variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Single credential for all AI calls. OpenRouter is the gateway across
# providers, so this is the only AI-related secret the app needs.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Default AI model — override via the AI_MODEL env var to use a different
# model. Must support OpenAI-compatible tool-calling (the chat loop
# depends on it); see AI_FALLBACK_MODELS below for the chain used when
# this one fails or produces degenerate output.
AI_MODEL = os.getenv("AI_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

# Fallback model chain — tried in order if the primary model fails or
# returns a degenerate output. OpenRouter's free-tier roster rotates over
# time, so a single model can become unavailable without warning; this
# chain gives the app a chance to recover rather than fail outright.
# Override via the AI_FALLBACK_MODELS env var (comma-separated list).
#
# Note: verify each slug against OpenRouter's current model list — the
# :free roster rotates. If a slug 404s at runtime, the chain just moves
# to the next model, so a stale entry here is not fatal.
_FALLBACK_MODELS_RAW = os.getenv("AI_FALLBACK_MODELS", "")
AI_FALLBACK_MODELS = [
    model.strip() for model in _FALLBACK_MODELS_RAW.split(",") if model.strip()
] or [
    "nvidia/nemotron-3.5-lightning:free",
    "inclusionai/ling-3.0-flash-fin:free",
    "qwen/qwen3.8-27b:free",
]

# OpenRouter's recommended optional headers, used for routing context and
# abuse/cost monitoring on their end. Placeholder values — update if/when
# JimmyCore has a real deployed URL.
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "JimmyCore")
OPENROUTER_APP_URL = os.getenv("OPENROUTER_APP_URL", "https://jimmycore.app")