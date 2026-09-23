from dotenv import load_dotenv
import os

load_dotenv()

# Configuration variables
DATABASE_URL = os.getenv("DATABASE_URL")

# OpenRouter configuration. Repurposed from the old Gemini-only AI_API_KEY —
# OPENROUTER_API_KEY is the single credential for all AI calls now that
# OpenRouter is the gateway across providers.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Default AI model — change this env var to use a different model.
# Supported free-tier models (as of mid-2026):
#   - meta-llama/llama-3.1-8b-instruct:free (recommended fallback)
#   - meta-llama/llama-3.3-70b-instruct:free (original default)
#   - other models may be available — set AI_MODEL env var to your preferred slug.
AI_MODEL = os.getenv("AI_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

# Fallback model chain — tried in order if the primary model fails or
# returns a degenerate output. OpenRouter's free-tier roster rotates over
# time, so a single model can become unavailable without warning; this
# chain gives the app a chance to recover rather than fail outright.
# Override via the AI_FALLBACK_MODELS env var (comma-separated list).
_FALLBACK_MODELS_RAW = os.getenv("AI_FALLBACK_MODELS", "")
AI_FALLBACK_MODELS = [
    model.strip() for model in _FALLBACK_MODELS_RAW.split(",") if model.strip()
] or [
    "nvidia/nemotron-3.5-lightning:free",
    "inclusionai/ling-3.0-flash-fin:free",
    "thinkingmachines/inkling-small:free",

]

# OpenRouter's recommended optional headers, used for routing context and
# abuse/cost monitoring on their end. Placeholder values — update if/when
# JimmyCore has a real deployed URL.
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "JimmyCore")
OPENROUTER_APP_URL = os.getenv("OPENROUTER_APP_URL", "https://jimmycore.app")