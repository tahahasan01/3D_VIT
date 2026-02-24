"""Garment processing API routes.

Exposes endpoints for converting 2D outfit images into
textured 3D garment meshes.
"""

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ..config import ALLOWED_IMAGE_TYPES, MAX_UPLOAD_SIZE_MB
from ..models.garment import GarmentMeasurements
from ..services.garment_processor import process_garment

router = APIRouter(prefix="/garment", tags=["garment"])


@router.post(
    "/process",
    response_class=Response,
    responses={
        200: {
            "content": {"model/gltf-binary": {}},
            "description": "Textured 3D garment model in GLB format.",
        },
    },
    summary="Convert 2D outfit image to 3D garment",
)
async def process_garment_image(
    image: UploadFile = File(
        ..., description="2D photograph of the outfit"
    ),
    measurements: str = Form(
        ..., description="JSON string of GarmentMeasurements"
    ),
) -> Response:
    """Accept an outfit image and sizing data, return a textured GLB."""

    # -- Validate content type ------------------------------------------------
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image type '{image.content_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}."
            ),
        )

    # -- Read and validate size -----------------------------------------------
    image_bytes = await image.read()
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Image exceeds {MAX_UPLOAD_SIZE_MB} MB limit.",
        )

    # -- Parse measurements JSON ----------------------------------------------
    try:
        garment_measurements = GarmentMeasurements(**json.loads(measurements))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid measurements JSON: {exc}",
        ) from exc

    # -- Process --------------------------------------------------------------
    try:
        glb_data, used_conforming, _vertex_map = process_garment(
            image_bytes,
            garment_measurements,
            body_landmarks=garment_measurements.body_landmarks,
            body_measurements=garment_measurements.body_measurements,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Garment processing failed: {exc}",
        ) from exc

    fit_value = "conforming" if used_conforming else "parametric"
    return Response(
        content=glb_data,
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": 'attachment; filename="garment.glb"',
            "X-Garment-Fit": fit_value,
        },
    )
