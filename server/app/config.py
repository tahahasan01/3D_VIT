"""Application configuration.

Centralizes all runtime settings, directory paths, and environment
variables used throughout the backend application.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure writable directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Server settings
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173"
).split(",")

# ---------------------------------------------------------------------------
# Upload constraints
# ---------------------------------------------------------------------------
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
ALLOWED_IMAGE_TYPES: set[str] = {"image/jpeg", "image/png", "image/webp"}

# ---------------------------------------------------------------------------
# FabricDiffusion (optional PBR texture rectification)
# ---------------------------------------------------------------------------
USE_FABRIC_DIFFUSION: bool = os.getenv("USE_FABRIC_DIFFUSION", "0").strip().lower() in ("1", "true", "yes")
FABRIC_DIFFUSION_ROOT: str | None = os.getenv("FABRIC_DIFFUSION_ROOT", "").strip() or None
FABRIC_DIFFUSION_PYTHON: str | None = os.getenv("FABRIC_DIFFUSION_PYTHON", "").strip() or None
FABRIC_DIFFUSION_TIMEOUT: int = int(os.getenv("FABRIC_DIFFUSION_TIMEOUT", "120"))
