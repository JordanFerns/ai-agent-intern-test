"""Application Configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledge-base"
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = BASE_DIR / "evaluation"
ORDERS_FILE = DATA_DIR / "orders.json"
CACHE_DIR = BASE_DIR / ".cache"

# Model and API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Agent Settings
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() in ("1", "true", "yes")

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True)
