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