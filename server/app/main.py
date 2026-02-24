"""FastAPI application entry point.

Creates and configures the main FastAPI application instance,
including CORS middleware and API router registration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.router import api_router
from .config import CORS_ORIGINS

app = FastAPI(
    title="VIT - Virtual Interactive Try-on API",
    description=(
        "API for generating 3D human models from body measurements "
        "and converting 2D outfit images into 3D garments for virtual try-on."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Body-Landmarks", "X-Body-Model-Type"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(api_router, prefix="/api")


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Return service health status."""
    return {"status": "healthy", "version": "0.1.0"}
