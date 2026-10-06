import os


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = env_int("DB_PORT", 5432)
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpassword")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = env_int("OLLAMA_PORT", 11434)
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:4b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:4b")
OCR_MODEL = os.getenv("OCR_MODEL", "deepseek-ocr")

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "/documents")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/rag-extraction")

TOP_K = env_int("TOP_K", 5)
OLLAMA_TIMEOUT = env_int("OLLAMA_TIMEOUT", 300)
PARENT_CHUNK_SIZE = 1200
CHILD_CHUNK_SIZE = 350
CHILD_CHUNK_OVERLAP = 50
