import os
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/agent_core/llm/settings.py -> repo root
load_dotenv(dotenv_path=REPO_ROOT / ".env")

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "local")
CHAT_MODEL      = os.getenv("CHAT_MODEL")

DEFAULT_TEMP = float(os.getenv("LLM_TEMP", 0.2))

if not OPENAI_API_BASE:
    raise RuntimeError("OPENAI_API_BASE is not set. Check your .env")
if not CHAT_MODEL:
    raise RuntimeError("CHAT_MODEL is not set. Check your .env")
