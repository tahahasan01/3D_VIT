"""ISP API: garment generation + draping via Implicit Sewing Patterns.

POST /isp/generate  — Generate a T-pose garment mesh (no draping)
POST /isp/drape     — Generate + drape a garment on a posed SMPL body
GET  /isp/status    — Check ISP readiness (checkpoints available, etc.)
"""

from __future__ import annotations

import base64
import json
import logging

from fastapi import APIRouter, Form, HTTPException

from ..services.isp_service import (
    GarmentKind,
    get_isp_service,
    is_isp_available,
    _check_checkpoints_available,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/isp", tags=["isp"])


@router.get("/status")
async def isp_status() -> dict:
    """Check ISP readiness — whether checkpoints and torch are available."""
    available_tee = is_isp_available("tee")
    available_pants = is_isp_available("pants")
    available_skirt = is_isp_available("skirt")
    missing = _check_checkpoints_available(None)
    try:
        import torch
        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
    except ImportError:
        torch_version = None
        cuda = False

    return {
        "isp_available": available_tee,  # at least tee
        "tee": available_tee,
        "pants": available_pants,
        "skirt": available_skirt,
        "missing_checkpoints": missing,
        "torch_version": torch_version,
        "cuda_available": cuda,
    }


@router.post("/generate")
async def isp_generate(
    garment_type: str = Form("tee", description="Garment type: tee, pants, or skirt"),
    idx_G: int = Form(0, description="Index into the learned garment codebook"),
    resolution: int = Form(180, description="UV grid resolution"),
) -> dict:
    """Generate a T-pose garment mesh using ISP (no draping).

    Returns the garment as a base64-encoded GLB.
    """
    kind = _validate_garment_kind(garment_type)
    if not is_isp_available(kind):
        missing = _check_checkpoints_available(kind)
        raise HTTPException(
            status_code=503,
            detail=f"ISP not available for '{kind}'. Missing checkpoints: {missing}. "
            f"Download from https://drive.google.com/file/d/1Zhr93ejWGobqDnJjE-P95ssNTDYSFNXS/view",
        )

    try:
        isp = get_isp_service()
        n_garments = isp.get_num_garments(kind)
        if idx_G < 0 or idx_G >= n_garments:
            raise HTTPException(
                status_code=400,
                detail=f"idx_G must be in [0, {n_garments - 1}] for '{kind}'. Got {idx_G}.",
            )
        mesh = isp.generate_tpose_garment(kind, idx_G, resolution)
        glb = mesh.export(file_type="glb")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ISP generation failed")
        raise HTTPException(status_code=500, detail=f"ISP generation failed: {exc}") from exc

    return {
        "success": True,
        "garment_glb_base64": base64.standard_b64encode(glb).decode("ascii"),
        "garment_type": kind,
        "idx_G": idx_G,
    }


@router.post("/drape")
async def isp_drape(
    garment_type: str = Form("tee", description="Garment type: tee, pants, or skirt"),
    idx_G: int = Form(0, description="Index into the learned garment codebook"),
    pose: str = Form("null", description="JSON array of 72 floats (SMPL pose axis-angle). null = T-pose."),
    beta: str = Form("null", description="JSON array of 10 floats (SMPL shape params). null = mean shape."),
    resolution: int = Form(180, description="UV grid resolution"),
    smooth: bool = Form(True, description="Apply Taubin smoothing"),
) -> dict:
    """Generate an ISP garment and drape it on a posed SMPL body.

    Returns both garment and body as base64-encoded GLBs.
    """
    import numpy as np

    kind = _validate_garment_kind(garment_type)
    if not is_isp_available(kind):
        missing = _check_checkpoints_available(kind)
        raise HTTPException(
            status_code=503,
            detail=f"ISP not available for '{kind}'. Missing: {missing}",
        )

    # Parse pose & beta
    pose_arr = _parse_json_array(pose, 72, "pose")
    beta_arr = _parse_json_array(beta, 10, "beta")

    try:
        isp = get_isp_service()
        n_garments = isp.get_num_garments(kind)
        if idx_G < 0 or idx_G >= n_garments:
            raise HTTPException(
                status_code=400,
                detail=f"idx_G must be in [0, {n_garments - 1}] for '{kind}'. Got {idx_G}.",
            )
        garment_mesh, body_mesh = isp.drape_garment(
            kind, idx_G,
            pose=pose_arr, beta=beta_arr,
            resolution=resolution, smooth=smooth,
        )
        garment_glb = garment_mesh.export(file_type="glb")
        body_glb = body_mesh.export(file_type="glb")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ISP draping failed")
        raise HTTPException(status_code=500, detail=f"ISP draping failed: {exc}") from exc

    return {
        "success": True,
        "garment_glb_base64": base64.standard_b64encode(garment_glb).decode("ascii"),
        "body_glb_base64": base64.standard_b64encode(body_glb).decode("ascii"),
        "garment_type": kind,
        "idx_G": idx_G,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_garment_kind(value: str) -> GarmentKind:
    v = value.strip().lower()
    if v in ("tee", "tshirt", "t-shirt", "shirt"):
        return "tee"
    if v in ("pants", "trousers"):
        return "pants"
    if v in ("skirt",):
        return "skirt"
    raise HTTPException(status_code=400, detail=f"Unknown garment type '{value}'. Use: tee, pants, skirt.")


def _parse_json_array(raw: str, expected_len: int, name: str):
    """Parse a JSON string into a numpy array or return None."""
    import numpy as np

    if raw.strip().lower() in ("null", "none", ""):
        return None
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {name} JSON: {exc}") from exc
    if not isinstance(arr, list):
        raise HTTPException(status_code=422, detail=f"{name} must be a JSON array of {expected_len} floats.")
    if len(arr) != expected_len:
        raise HTTPException(status_code=422, detail=f"{name} must have {expected_len} values, got {len(arr)}.")
    return np.array(arr, dtype=np.float32)
