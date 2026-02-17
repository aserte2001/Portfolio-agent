"""
Configuration module for AI Portfolio Agent.
Loads environment variables and provides project-wide constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Configuration ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# ── Application Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_PATH = os.path.join(BASE_DIR, "portfolio.json")
PROFILE_PATH = os.path.join(BASE_DIR, "investment_profile.json")
CHAT_HISTORY_PATH = os.path.join(BASE_DIR, "chat_history.json")

# ── Agent Defaults ─────────────────────────────────────────────────
AGENT_VERBOSE = True
MAX_ITERATIONS = 15
REQUEST_TIMEOUT = 120
MAX_CHAT_MESSAGES = 50

# ── UI Constants ───────────────────────────────────────────────────
APP_TITLE = "AI Portfolio Agent"
APP_ICON = "\U0001f916"
SUPPORTED_MODELS = ["gpt-4o", "gpt-4o-mini"]
