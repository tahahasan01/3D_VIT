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


def _body_skin_color_factor(body_m: BodyMeasurements) -> list[float] | None:
    """Return [r, g, b, 1] from body_m.skin_color_hex if set, else None (use default)."""
    hex_str = getattr(body_m, "skin_color_hex", None)
    if not hex_str or not isinstance(hex_str, str):
        return None
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) != 6:
        return None
    try:
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        return [r, g, b, 1.0]
    except ValueError:
        return None


@router.post("/create")
async def create_tryon(
    garment_image: UploadFile = File(..., description="2D photograph of the outfit (flat garment image)"),
    additional_images: list[UploadFile] = File(default=[], description="Optional extra angle photos (back, side, detail)"),
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

    additional_image_bytes: list[bytes] = []
    for extra in additional_images:
        if extra.content_type not in ALLOWED_IMAGE_TYPES:
            logger.warning("Skipping additional image with unsupported type: %s", extra.content_type)
            continue
        data = await extra.read()
        if len(data) > max_bytes:
            logger.warning("Skipping additional image that exceeds size limit")
            continue
        additional_image_bytes.append(data)

    if additional_image_bytes:
        logger.info("Received %d additional angle image(s)", len(additional_image_bytes))

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

    # Step 2: Build skinned body GLB (with face texture and skin color when SMPL)
    try:
        if skinning_data is not None:
            from ..services.skinned_glb_builder import build_skinned_glb
            base_color = _body_skin_color_factor(body_m)
            body_glb = build_skinned_glb(
                body_mesh,
                skinning_data["weights"],
                skinning_data["kintree_parents"],
                skinning_data["joints_positions"],
                head_face_indices=skinning_data.get("head_face_indices"),
                base_color_factor=base_color,
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
            additional_images=additional_image_bytes or None,
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
