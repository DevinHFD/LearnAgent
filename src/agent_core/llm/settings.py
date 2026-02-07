import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "local")
CHAT_MODEL      = os.getenv("CHAT_MODEL")

DEFAULT_TEMP = float(os.getenv("LLM_TEMP", 0.2))
