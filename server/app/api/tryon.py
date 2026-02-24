"""Try-on API: combined body + garment pipeline.

Parametric approach: body from measurements, garment from template + texture.
POST /create: body from measurements + garment from 2D image (conforming/parametric).
"""

import base64
import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import ALLOWED_IMAGE_TYPES, MAX_UPLOAD_SIZE_MB
from ..models.body import BodyMeasurements
from ..models.garment import GarmentMeasurements
from ..services.body_generator import get_body_mesh
from ..services.garment_processor import process_garment

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tryon", tags=["tryon"])


@router.post("/create")
async def create_tryon(
    garment_image: UploadFile = File(..., description="2D photograph of the outfit (flat garment image)"),
    body_measurements: str = Form(..., description="JSON string of BodyMeasurements"),
    garment_measurements: str = Form(..., description="JSON string of GarmentMeasurements"),
) -> dict:
    """
    Full pipeline: generate body from measurements, then process garment image
    with that body. Garment is built as conforming offset surface (when SMPL body)
    or parametric template, textured from the uploaded image.
    Returns body GLB and garment GLB (both skinned+animated when SMPL).
    """
    if garment_image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{garment_image.content_type}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}.",
        )

    image_bytes = await garment_image.read()
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Image exceeds {MAX_UPLOAD_SIZE_MB} MB limit.",
        )

    try:
        body_m = BodyMeasurements(**json.loads(body_measurements))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid body_measurements JSON: {exc}") from exc

    try:
        garment_m_dict = json.loads(garment_measurements)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid garment_measurements JSON: {exc}") from exc

    # Step 1: Generate body mesh, landmarks, and skinning data
    try:
        body_mesh, landmarks, skinning_data = get_body_mesh(body_m)
    except Exception as exc:
        logger.exception("Body generation failed")
        raise HTTPException(status_code=500, detail=f"Body generation failed: {exc}") from exc

    # Step 2: Build skinned body GLB
    try:
        if skinning_data is not None:
            from ..services.skinned_glb_builder import build_skinned_glb
            body_glb = build_skinned_glb(
                body_mesh,
                skinning_data["weights"],
                skinning_data["kintree_parents"],
                skinning_data["joints_positions"],
            )
        else:
            body_glb = body_mesh.export(file_type="glb")
    except Exception as exc:
        logger.exception("Body GLB export failed")
        raise HTTPException(status_code=500, detail=f"Body GLB export failed: {exc}") from exc

    # Step 3: Process garment with this body (conforming when body available)
    garment_m_dict["body_measurements"] = (
        body_m.model_dump() if hasattr(body_m, "model_dump") else body_m.dict()
    )
    if landmarks:
        garment_m_dict["body_landmarks"] = landmarks
    garment_m = GarmentMeasurements(**garment_m_dict)

    try:
        garment_glb, used_conforming, vertex_map = process_garment(
            image_bytes,
            garment_m,
            body_landmarks=landmarks or {},
            body_measurements=body_m.model_dump(),
            body_mesh=body_mesh,
            height_m=body_m.height_cm / 100.0,
        )
    except Exception as exc:
        logger.exception("Garment processing failed")
        raise HTTPException(status_code=500, detail=f"Garment processing failed: {exc}") from exc

    # Step 4: If skinning data and vertex map available, rebuild garment GLB as skinned
    if skinning_data is not None and vertex_map is not None:
        try:
            from ..services.skinned_glb_builder import build_skinned_garment_glb
            garment_glb = build_skinned_garment_glb(
                garment_glb,  # raw GLB bytes from process_garment (we'll reload the mesh)
                skinning_data["weights"],
                vertex_map,
                skinning_data["kintree_parents"],
                skinning_data["joints_positions"],
            )
        except Exception as exc:
            logger.warning("Skinned garment GLB failed, using static: %s", exc)

    return {
        "success": True,
        "body_glb_base64": base64.standard_b64encode(body_glb).decode("ascii"),
        "garment_glb_base64": base64.standard_b64encode(garment_glb).decode("ascii"),
        "body_landmarks": landmarks,
        "garment_fit": "isp" if used_conforming else "parametric",
        "body_model_type": "smpl" if skinning_data is not None else "parametric",
    }
