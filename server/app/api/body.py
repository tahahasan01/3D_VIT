"""Body generation API routes.

Exposes endpoints for creating 3D human body meshes from
a set of body measurements.
"""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ..models.body import BodyMeasurements
from ..services.body_generator import generate_body

router = APIRouter(prefix="/body", tags=["body"])


@router.post(
    "/generate",
    response_class=Response,
    responses={
        200: {
            "content": {"model/gltf-binary": {}},
            "description": "Generated 3D body model in GLB format (skinned+animated when SMPL).",
        },
    },
    summary="Generate 3D body from measurements",
)
async def generate_body_model(measurements: BodyMeasurements) -> Response:
    """Accept body measurements and return a GLB binary of the 3D body.

    When SMPL body is used, the GLB includes a 24-joint skeleton with
    walk and twirl animations. The response includes an ``X-Body-Landmarks``
    header containing JSON-encoded body landmark positions for garment fitting.
    """
    try:
        glb_data, landmarks, model_type = generate_body(measurements)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Body generation failed: {exc}",
        ) from exc

    headers: dict[str, str] = {
        "Content-Disposition": 'attachment; filename="body.glb"',
        "X-Body-Model-Type": model_type,
    }
    if landmarks is not None:
        headers["X-Body-Landmarks"] = json.dumps(landmarks)

    return Response(
        content=glb_data,
        media_type="model/gltf-binary",
        headers=headers,
    )
