"""
Configuration module for the entire application
All environment variables and constants defined here
Used by: ALL team members

This file is intentionally tolerant when API keys are missing so unit
tests can import the module without requiring real credentials. It
exposes `API_PROVIDER` and `API_KEY` which the adapter imports to choose
between Gemini/OpenAI or a local stub.
"""
import os
from dotenv import load_dotenv
import pathlib

# Load environment variables from .env file (if present)
load_dotenv()

# ============= PROVIDER KEYS =============
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Determine which provider to use. Priority: GEMINI -> OPENAI -> stub
if GEMINI_API_KEY:
    API_PROVIDER = "gemini"
    API_KEY = GEMINI_API_KEY
elif OPENAI_API_KEY:
    API_PROVIDER = "openai"
    API_KEY = OPENAI_API_KEY
else:
    API_PROVIDER = "stub"
    API_KEY = None

# ============= MODEL / TUNING DEFAULTS =============
# Keep defaults so code can import these even without env vars set
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))

# ============= CHROMADB CONFIGURATION =============
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "fda_drug_labels")

# ============= DATA PATHS =============
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = os.getenv("DATA_PATH", str(ROOT / "data"))
FDA_LABELS_FILE = os.path.join(DATA_PATH, "fda_labels.csv")
CHUNKS_FILE = os.path.join(DATA_PATH, "fda_chunks.json")
EMBEDDINGS_FILE = os.path.join(DATA_PATH, "fda_chunks_with_embeddings.json")

# ============= RETRIEVAL PARAMETERS =============
DEFAULT_N_RESULTS = 10
RERANK_N_RESULTS = 5
EMBEDDING_DIMENSION = 1536

# ============= LOGGING =============
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

if __name__ == "__main__":
    print("✓ Configuration loaded successfully")
    print(f"  - API_PROVIDER: {API_PROVIDER}")
    print(f"  - EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"  - LLM_MODEL: {LLM_MODEL}")
    print(f"  - CHROMA_DB_PATH: {CHROMA_DB_PATH}")
