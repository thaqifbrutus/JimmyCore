from dotenv import load_dotenv
import os

load_dotenv()

# Configuration variables
DATABASE_URL = os.getenv("DATABASE_URL")

# OpenRouter configuration. Repurposed from the old Gemini-only AI_API_KEY —
# OPENROUTER_API_KEY is the single credential for all AI calls now that
# OpenRouter is the gateway across providers.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Single model slug used by default for all AI calls. Parameterized at the
# call site (see ai_service._call_ai_model) so a different model could be
# passed later without a refactor, even though nothing does that yet.
#
# meta-llama/llama-3.3-70b-instruct:free is used as the default here — it's
# a long-standing, stable OpenRouter free-tier slug (available since the
# model's release, still listed free as of mid-2026). OpenRouter's free
# roster does rotate over time, so this should be revisited if the env var
# ever returns a "model not found" error from OpenRouter — check
# https://openrouter.ai/models for the current free-tier list and update
# the AI_MODEL env var (no code change needed).
AI_MODEL = os.getenv("AI_MODEL", "nex-agi/nex-n2-pro:free")

# OpenRouter's recommended optional headers, used for routing context and
# abuse/cost monitoring on their end. Placeholder values — update if/when
# JimmyCore has a real deployed URL.
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "JimmyCore")
OPENROUTER_APP_URL = os.getenv("OPENROUTER_APP_URL", "https://jimmycore.app")