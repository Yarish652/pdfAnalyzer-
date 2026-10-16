import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3.5-lightning:free",
)


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default

    return value if value > 0 else default


MAX_UPLOAD_SIZE_BYTES = _positive_int_env("MAX_UPLOAD_SIZE_BYTES", 20 * 1024 * 1024)
MAX_PDF_PAGES = _positive_int_env("MAX_PDF_PAGES", 300)

# Read pages that have no text layer (scans, image-only slides) with OCR.
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}

# Sentence embedding model for chunks and questions. Changing it starts a new
# vector collection, so documents must be uploaded again.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL
