import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "8"))


def embedding_dim() -> int:
    """Vector dimension, inferred from the provider/model when not set explicitly."""
    explicit = os.getenv("EMBEDDING_DIM", "").strip()
    if explicit:
        return int(explicit)
    if EMBEDDING_PROVIDER == "openai":
        return 1536
    if EMBEDDING_PROVIDER == "hash":
        return 512
    return 384


EMBEDDING_DIM = embedding_dim()
