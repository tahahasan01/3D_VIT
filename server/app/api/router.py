"""Top-level API router.

Aggregates all feature routers under the ``/api`` prefix.
"""

from fastapi import APIRouter

from .body import router as body_router
from .garment import router as garment_router
from .isp import router as isp_router
from .tryon import router as tryon_router

api_router = APIRouter()
api_router.include_router(body_router)
api_router.include_router(garment_router)
api_router.include_router(isp_router)
api_router.include_router(tryon_router)
