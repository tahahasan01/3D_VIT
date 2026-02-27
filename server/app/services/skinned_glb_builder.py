"""Skinned GLB builder — constructs glTF binary with skeleton + animations.

Builds a compliant glTF 2.0 binary (.glb) that includes:
- A 24-joint SMPL skeleton (node hierarchy)
- Inverse bind matrices for each joint
- Per-vertex skinning weights (JOINTS_0 + WEIGHTS_0 attributes)
- Walk animation clip (~1.2 s cycle)
- Twirl animation clip (~2.0 s, 360-degree Y rotation)
- PBR material with optional base color, vertex colors, or texture

Uses pygltflib for glTF construction and scipy for quaternion math.
"""

from __future__ import annotations

import io
import json
import logging
import struct
from typing import Sequence

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# GLB chunk type constants
_GLB_CHUNK_JSON = 0x4E4F534A
_GLB_CHUNK_BIN = 0x004E4942

# SMPL joint names (must match smpl_body._JOINT_NAMES)
_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]

# Default body color — neutral gray matching the reference SMPL clay look
_SKIN_COLOR_FACTOR = [200 / 255, 200 / 255, 200 / 255, 1.0]


def _axis_angle_to_quat(axis: Sequence[float], angle_deg: float) -> list[float]:
    """Convert axis-angle to quaternion [x, y, z, w] (glTF convention)."""
    angle_rad = np.radians(angle_deg)
    axis_np = np.array(axis, dtype=np.float64)
    axis_np = axis_np / (np.linalg.norm(axis_np) + 1e-12)
    rotvec = axis_np * angle_rad
    r = Rotation.from_rotvec(rotvec)
    q = r.as_quat()  # [x, y, z, w] — scipy default, matches glTF
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _identity_quat() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0]


def _walk_keyframes() -> dict[int, list[list[float]]]:
    """Generate walk animation keyframes for relevant joints.

    Returns dict mapping joint_index -> list of 4 quaternions (one per keyframe).
    Walk cycle: right-forward -> passing -> left-forward -> passing -> loop
    """
    keyframes: dict[int, list[list[float]]] = {}

    # Left hip (1): swings forward/back around X
    keyframes[1] = [
        _axis_angle_to_quat([1, 0, 0], -25),   # back
        _axis_angle_to_quat([1, 0, 0], 0),      # passing
        _axis_angle_to_quat([1, 0, 0], 25),     # forward
        _axis_angle_to_quat([1, 0, 0], 0),      # passing
    ]
    # Right hip (2): opposite phase
    keyframes[2] = [
        _axis_angle_to_quat([1, 0, 0], 25),
        _axis_angle_to_quat([1, 0, 0], 0),
        _axis_angle_to_quat([1, 0, 0], -25),
        _axis_angle_to_quat([1, 0, 0], 0),
    ]
    # Left knee (4): only flexes forward (positive X rotation)
    keyframes[4] = [
        _axis_angle_to_quat([1, 0, 0], 50),
        _axis_angle_to_quat([1, 0, 0], 5),
        _axis_angle_to_quat([1, 0, 0], 5),
        _axis_angle_to_quat([1, 0, 0], 5),
    ]
    # Right knee (5): opposite phase
    keyframes[5] = [
        _axis_angle_to_quat([1, 0, 0], 5),
        _axis_angle_to_quat([1, 0, 0], 5),
        _axis_angle_to_quat([1, 0, 0], 50),
        _axis_angle_to_quat([1, 0, 0], 5),
    ]
    # Left ankle (7)
    keyframes[7] = [
        _axis_angle_to_quat([1, 0, 0], 10),
        _axis_angle_to_quat([1, 0, 0], -5),
        _axis_angle_to_quat([1, 0, 0], -10),
        _axis_angle_to_quat([1, 0, 0], -5),
    ]
    # Right ankle (8)
    keyframes[8] = [
        _axis_angle_to_quat([1, 0, 0], -10),
        _axis_angle_to_quat([1, 0, 0], -5),
        _axis_angle_to_quat([1, 0, 0], 10),
        _axis_angle_to_quat([1, 0, 0], -5),
    ]
    # Left shoulder (16): counter-swing
    keyframes[16] = [
        _axis_angle_to_quat([1, 0, 0], 15),
        _axis_angle_to_quat([1, 0, 0], 0),
        _axis_angle_to_quat([1, 0, 0], -15),
        _axis_angle_to_quat([1, 0, 0], 0),
    ]
    # Right shoulder (17): opposite
    keyframes[17] = [
        _axis_angle_to_quat([1, 0, 0], -15),
        _axis_angle_to_quat([1, 0, 0], 0),
        _axis_angle_to_quat([1, 0, 0], 15),
        _axis_angle_to_quat([1, 0, 0], 0),
    ]
    # Left elbow (18): slight flex
    keyframes[18] = [
        _axis_angle_to_quat([1, 0, 0], 15),
        _axis_angle_to_quat([1, 0, 0], 5),
        _axis_angle_to_quat([1, 0, 0], 5),
        _axis_angle_to_quat([1, 0, 0], 5),
    ]
    # Right elbow (19)
    keyframes[19] = [
        _axis_angle_to_quat([1, 0, 0], 5),
        _axis_angle_to_quat([1, 0, 0], 5),
        _axis_angle_to_quat([1, 0, 0], 15),
        _axis_angle_to_quat([1, 0, 0], 5),
    ]

    return keyframes


def _twirl_keyframes() -> dict[int, list[list[float]]]:
    """Generate twirl animation keyframes — 360 degree Y rotation on root.

    Returns dict mapping joint_index -> list of 5 quaternions.
    """
    return {
        0: [
            _axis_angle_to_quat([0, 1, 0], 0),
            _axis_angle_to_quat([0, 1, 0], 90),
            _axis_angle_to_quat([0, 1, 0], 180),
            _axis_angle_to_quat([0, 1, 0], 270),
            _axis_angle_to_quat([0, 1, 0], 360),
        ]
    }


def _t_pose_keyframes() -> dict[int, list[list[float]]]:
    """Identity (T-pose) for joints that we animate in a_pose/natural_stand.

    Returns dict mapping joint_index -> list of 2 identical identity quaternions.
    """
    identity = _identity_quat()
    return {
        4: [identity, identity],
        5: [identity, identity],
        12: [identity, identity],
        13: [identity, identity],
        14: [identity, identity],
        16: [identity, identity],
        17: [identity, identity],
        18: [identity, identity],
        19: [identity, identity],
    }


def _a_pose_keyframes() -> dict[int, list[list[float]]]:
    """Static A-pose: arms lowered ~30 deg from T-pose, slight elbow bend.

    Returns dict mapping joint_index -> list of 2 identical quaternions (static clip).
    """
    # Shoulders: lower arms (Z rotation — sign so arms go downward from T-pose)
    left_shoulder_q = _axis_angle_to_quat([0, 0, 1], -30)
    right_shoulder_q = _axis_angle_to_quat([0, 0, 1], 30)
    # Elbows: slight bend (X rotation)
    elbow_q = _axis_angle_to_quat([1, 0, 0], 10)
    return {
        16: [left_shoulder_q, left_shoulder_q],
        17: [right_shoulder_q, right_shoulder_q],
        18: [elbow_q, elbow_q],
        19: [elbow_q, elbow_q],
    }


def _natural_stand_keyframes() -> dict[int, list[list[float]]]:
    """Natural relaxed standing: arms at sides, slight elbow/knee bend, relaxed neck.

    Returns dict mapping joint_index -> list of 2 identical quaternions (static clip).
    """
    # Arms at sides: shoulders Z rotation so arms hang downward from T-pose
    left_shoulder_q = _axis_angle_to_quat([0, 0, 1], -65)
    right_shoulder_q = _axis_angle_to_quat([0, 0, 1], 65)
    # Elbows: slight bend
    elbow_q = _axis_angle_to_quat([1, 0, 0], 25)
    # Collars: minimal
    left_collar_q = _axis_angle_to_quat([0, 0, 1], 5)
    right_collar_q = _axis_angle_to_quat([0, 0, 1], -5)
    # Knees: micro-bend
    knee_q = _axis_angle_to_quat([1, 0, 0], 2)
    # Neck: slight tilt down
    neck_q = _axis_angle_to_quat([1, 0, 0], -3)
    return {
        12: [neck_q, neck_q],
        13: [left_collar_q, left_collar_q],
        14: [right_collar_q, right_collar_q],
        16: [left_shoulder_q, left_shoulder_q],
        17: [right_shoulder_q, right_shoulder_q],
        18: [elbow_q, elbow_q],
        19: [elbow_q, elbow_q],
        4: [knee_q, knee_q],
        5: [knee_q, knee_q],
    }


def _extract_material_from_mesh(mesh: trimesh.Trimesh) -> dict:
    """Extract material info from a trimesh for embedding in glTF.

    Returns a dict with one of:
    - {'type': 'color', 'base_color_factor': [r,g,b,a]}  (0-1 floats)
    - {'type': 'vertex_colors', 'colors': np.ndarray}    (N,4 float32 0-1)
    - {'type': 'texture', 'png_bytes': bytes, 'uv': np.ndarray}
    - {'type': 'none'}
    """
    visual = mesh.visual
    if visual is None:
        return {"type": "none"}

    # Check for vertex colors (ColorVisuals)
    if isinstance(visual, trimesh.visual.ColorVisuals):
        vc = visual.vertex_colors
        if vc is not None and len(vc) == len(mesh.vertices):
            vc = np.asarray(vc, dtype=np.float32)
            if vc.max() > 1.0:
                vc = vc / 255.0
            if vc.shape[1] == 3:
                vc = np.hstack([vc, np.ones((len(vc), 1), dtype=np.float32)])
            # If all vertex colors are the same, use base_color_factor instead
            if np.allclose(vc, vc[0]):
                return {"type": "color", "base_color_factor": vc[0].tolist()}
            return {"type": "vertex_colors", "colors": vc}

    # Check for texture (TextureVisuals)
    if isinstance(visual, trimesh.visual.TextureVisuals):
        uv = visual.uv
        mat = visual.material
        if uv is not None and len(uv) == len(mesh.vertices):
            # Try to get texture image
            img = None
            if hasattr(mat, "image") and mat.image is not None:
                img = mat.image
            elif hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                img = mat.baseColorTexture

            if img is not None:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                png_buf = io.BytesIO()
                img.convert("RGB").save(png_buf, format="PNG")
                return {
                    "type": "texture",
                    "png_bytes": png_buf.getvalue(),
                    "uv": np.asarray(uv, dtype=np.float32),
                }

            # Has UVs but no texture image — check for base color on material
            if hasattr(mat, "main_color") and mat.main_color is not None:
                c = np.asarray(mat.main_color, dtype=np.float32)
                if c.max() > 1.0:
                    c = c / 255.0
                if len(c) == 3:
                    c = np.append(c, 1.0)
                return {"type": "color", "base_color_factor": c.tolist()}

    return {"type": "none"}


def _extract_first_image_from_glb(glb_bytes: bytes) -> bytes | None:
    """Extract the first image's raw bytes from a GLB file.

    Trimesh often does not expose texture image on the mesh after loading a GLB
    (PBRMaterial.baseColorTexture is None). This parses the GLB structure
    and returns the embedded image bytes so we can re-embed them in the skinned GLB.

    Returns
    -------
    bytes or None
        PNG (or other) image bytes, or None if no image or parse error.
    """
    if len(glb_bytes) < 28:
        return None
    try:
        magic, version, total_len = struct.unpack_from("<III", glb_bytes, 0)
        if magic != 0x46546C67 or version != 2:
            return None
        # First chunk: length (4), type (4), data
        chunk0_len, chunk0_type = struct.unpack_from("<II", glb_bytes, 12)
        if chunk0_type != _GLB_CHUNK_JSON:
            return None
        json_start = 20
        json_end = json_start + chunk0_len
        if json_end > len(glb_bytes):
            return None
        doc = json.loads(glb_bytes[json_start:json_end].decode("utf-8"))
        images = doc.get("images") or []
        buffer_views = doc.get("bufferViews") or []
        buffers = doc.get("buffers") or []
        if not images or not buffer_views:
            return None
        first_image = images[0]
        bv_index = first_image.get("bufferView")
        if bv_index is None:
            return None
        bv = buffer_views[bv_index]
        bv_offset = bv.get("byteOffset", 0)
        bv_length = bv.get("byteLength", 0)
        # Second chunk is BIN
        bin_chunk_start = 12 + 8 + chunk0_len
        if bin_chunk_start + 8 > len(glb_bytes):
            return None
        bin_len, bin_type = struct.unpack_from("<II", glb_bytes, bin_chunk_start)
        if bin_type != _GLB_CHUNK_BIN:
            return None
        bin_data_start = bin_chunk_start + 8
        start = bin_data_start + bv_offset
        end = start + bv_length
        if end > len(glb_bytes) or start < bin_data_start:
            return None
        return glb_bytes[start:end]
    except (struct.error, json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug("Could not extract image from GLB: %s", e)
        return None


def _generate_face_texture_png(
    size: int = 1024,
    skin_rgb: tuple[int, int, int] = (200, 200, 200),
) -> bytes:
    """Generate a mannequin face texture with clearly visible features.

    UV convention (matching _generate_head_uvs):
      U=0.5 → face centre (horizontal)
      V=0   → top of head, V=1 → chin

    Returns PNG bytes suitable for embedding in a GLB.
    """
    brow_dark = (65, 45, 35)
    eye_white = (255, 252, 248)
    iris = (70, 50, 40)
    pupil = (20, 15, 10)
    lip_color = (175, 85, 75)
    lip_outline = (140, 65, 55)
    nose_shadow = (
        max(0, skin_rgb[0] - 30),
        max(0, skin_rgb[1] - 28),
        max(0, skin_rgb[2] - 18),
    )
    cheek_blush = (
        min(255, skin_rgb[0] + 12),
        max(0, skin_rgb[1] - 10),
        max(0, skin_rgb[2] - 12),
    )
    eyelid_shadow = (
        max(0, skin_rgb[0] - 18),
        max(0, skin_rgb[1] - 16),
        max(0, skin_rgb[2] - 12),
    )

    img = Image.new("RGB", (size, size), skin_rgb)
    draw = ImageDraw.Draw(img)

    cx = size // 2
    # Vertical positions calibrated from actual SMPL UV values (debug-verified):
    #   y_margin_above=0.20 → eye V≈0.356, nose V≈0.543, mouth V≈0.730, chin V≈0.993
    eye_y  = int(size * 0.355)
    nose_y = int(size * 0.540)
    lip_y  = int(size * 0.710)

    # Horizontal: cylindrical UV puts eye sockets at theta≈±25° from front,
    # → U_offset ≈ 0.071 from centre.  Use 0.073 for a tiny margin.
    eye_dx = int(size * 0.073)

    # --- Eyebrows ---
    brow_y  = int(size * 0.295)
    brow_hw = int(size * 0.055)
    brow_width = max(4, size // 64)
    for sign in (-1, 1):
        bx = cx + sign * eye_dx
        draw.arc(
            [bx - brow_hw, brow_y - 8, bx + brow_hw, brow_y + 8],
            180 if sign > 0 else 0,
            360 if sign > 0 else 180,
            fill=brow_dark,
            width=brow_width,
        )

    # --- Eyelid shadow ---
    lid_r = int(size * 0.048)
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        draw.ellipse(
            [ex - lid_r, eye_y - lid_r, ex + lid_r, eye_y + int(lid_r * 0.5)],
            fill=eyelid_shadow,
        )

    # --- Eyes (white sclera, iris, pupil) ---
    eye_rx = int(size * 0.042)
    eye_ry = int(size * 0.030)
    iris_r  = int(size * 0.022)
    pupil_r = int(size * 0.011)
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        draw.ellipse(
            [ex - eye_rx, eye_y - eye_ry, ex + eye_rx, eye_y + eye_ry],
            fill=eye_white, outline=(140, 120, 110), width=2,
        )
        draw.ellipse(
            [ex - iris_r, eye_y - iris_r, ex + iris_r, eye_y + iris_r],
            fill=iris,
        )
        draw.ellipse(
            [ex - pupil_r, eye_y - pupil_r, ex + pupil_r, eye_y + pupil_r],
            fill=pupil,
        )

    # --- Nose ---
    nose_w = max(4, size // 64)
    draw.line(
        [(cx, int(nose_y - size * 0.055)), (cx, int(nose_y + size * 0.025))],
        fill=nose_shadow, width=nose_w,
    )
    nostril_dx = int(size * 0.022)
    nostril_r  = int(size * 0.013)
    for sign in (-1, 1):
        nx = cx + sign * nostril_dx
        draw.ellipse(
            [nx - nostril_r, nose_y + int(size * 0.018),
             nx + nostril_r, nose_y + int(size * 0.040)],
            fill=nose_shadow,
        )

    # --- Cheek blush (subtle, wide) ---
    blush_r = int(size * 0.055)
    blush_y = int(size * 0.46)
    for sign in (-1, 1):
        bx = cx + sign * int(size * 0.12)
        draw.ellipse(
            [bx - blush_r, blush_y - blush_r // 2, bx + blush_r, blush_y + blush_r // 2],
            fill=cheek_blush,
        )

    # --- Lips ---
    lip_w       = int(size * 0.085)
    lip_h_upper = int(size * 0.016)
    lip_h_lower = int(size * 0.024)
    draw.ellipse(
        [cx - lip_w, lip_y - lip_h_upper, cx + lip_w, lip_y + lip_h_upper],
        fill=lip_color, outline=lip_outline, width=2,
    )
    draw.ellipse(
        [cx - int(lip_w * 0.9), lip_y, cx + int(lip_w * 0.9), lip_y + lip_h_lower * 2],
        fill=lip_color, outline=lip_outline, width=2,
    )
    draw.line(
        [(cx - lip_w + 4, lip_y), (cx + lip_w - 4, lip_y)],
        fill=lip_outline, width=max(2, size // 256),
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _generate_body_uvs(
    vertices: np.ndarray,
) -> np.ndarray:
    """Compute cylindrical UVs for the full body mesh.

    Uses the body's bounding box to normalise Y, and atan2(x, z) for the
    angular component so the front of the body maps to U≈0.5.

    Returns (n_verts, 2) float32 in [0, 1].
    """
    x = vertices[:, 0].astype(np.float64)
    y = vertices[:, 1].astype(np.float64)
    z = vertices[:, 2].astype(np.float64)

    # Angular U: atan2(x, z) so front (+Z) → U = 0.5
    theta = np.arctan2(x, z)
    u = 0.5 - theta / (2.0 * np.pi)

    # Vertical V: 0 at top, 1 at bottom (matching face texture convention)
    y_min, y_max = float(y.min()), float(y.max())
    y_range = max(y_max - y_min, 1e-6)
    v = 1.0 - (y - y_min) / y_range

    uvs = np.stack([u, v], axis=-1).astype(np.float32)
    return np.clip(uvs, 0.0, 1.0)


def _generate_head_uvs(
    vertices: np.ndarray,
    faces: np.ndarray,
    head_face_indices: np.ndarray,
    head_center: np.ndarray,
    joints_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Compute cylindrical (frontal) UVs for head vertices; others get (0, 0).

    Returns (n_verts, 2) float32 in [0, 1]. Head region mapped so the face
    front (max-Z side in SMPL) lands at texture centre (0.5, 0.5).
    """
    n_verts = len(vertices)
    uvs = np.zeros((n_verts, 2), dtype=np.float32)
    head_verts = np.unique(faces[head_face_indices].ravel())
    if len(head_verts) == 0:
        return uvs

    v = vertices[head_verts]
    x = v[:, 0] - head_center[0]
    y = v[:, 1] - head_center[1]
    z = v[:, 2] - head_center[2]

    front_sign = 1.0 if z.mean() >= 0 else -1.0

    theta = np.arctan2(x, z * front_sign)
    y_min, y_max = float(y.min()), float(y.max())
    y_range = max(y_max - y_min, 1e-6)
    v_norm = (y - y_min) / y_range

    u = 0.5 - theta / (2.0 * np.pi)
    v_coord = 1.0 - v_norm

    uvs[head_verts, 0] = np.clip(u, 0.0, 1.0).astype(np.float32)
    uvs[head_verts, 1] = np.clip(v_coord, 0.0, 1.0).astype(np.float32)

    # #region agent log
    import time as _t2, json as _j2
    _lp2 = r"c:\Users\Syed Taha Hasan\Desktop\Vit\debug-6b80da.log"
    log_data = {
        "head_center": head_center.tolist(),
        "y_min_rel": y_min, "y_max_rel": y_max, "y_range": y_range,
        "head_abs_y_min": float(v[:, 1].min() + head_center[1]),
        "head_abs_y_max": float(v[:, 1].max() + head_center[1]),
        "v_coord_min": float(v_coord.min()), "v_coord_max": float(v_coord.max()),
        "v_coord_mean": float(v_coord.mean()),
        "front_sign": front_sign,
    }
    if joints_positions is not None:
        head_j_y = float(joints_positions[15][1])
        neck_j_y = float(joints_positions[12][1])
        head_j_y_rel = head_j_y - float(head_center[1])
        neck_j_y_rel = neck_j_y - float(head_center[1])
        head_j_v = 1.0 - (head_j_y_rel - y_min) / y_range
        neck_j_v = 1.0 - (neck_j_y_rel - y_min) / y_range
        log_data["head_joint_abs_y"] = head_j_y
        log_data["neck_joint_abs_y"] = neck_j_y
        log_data["head_joint_v"] = head_j_v
        log_data["neck_joint_v"] = neck_j_v
        frontal_mask = (np.abs(theta) < np.pi / 4)
        if frontal_mask.sum() > 0:
            frontal_v = v_coord[frontal_mask]
            log_data["frontal_v_min"] = float(frontal_v.min())
            log_data["frontal_v_max"] = float(frontal_v.max())
            log_data["frontal_v_mean"] = float(frontal_v.mean())
    try:
        with open(_lp2, "a") as _f2:
            _f2.write(_j2.dumps({"sessionId":"6b80da","location":"_generate_head_uvs","message":"uv-mapping-details","data":log_data,"timestamp":int(_t2.time()*1000),"hypothesisId":"UV"})+"\n")
    except Exception:
        pass
    # #endregion

    return uvs


def build_skinned_glb(
    mesh: trimesh.Trimesh,
    weights: np.ndarray,
    kintree_parents: np.ndarray,
    joints_positions: np.ndarray,
    *,
    base_color_factor: list[float] | None = None,
    material_info: dict | None = None,
    head_face_indices: np.ndarray | None = None,
) -> bytes:
    """Build a skinned+animated GLB from mesh, skinning weights, skeleton.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Body mesh with vertices and faces.
    weights : np.ndarray, shape (N_verts, 24)
        Skinning weights per vertex per joint.
    kintree_parents : np.ndarray, shape (24,)
        Parent joint index for each joint (-1 for root).
    joints_positions : np.ndarray, shape (24, 3)
        Rest-pose joint positions in world space.
    base_color_factor : list[float], optional
        RGBA [0-1] base color for the mesh material. Defaults to skin color.
    material_info : dict, optional
        Material info from _extract_material_from_mesh(). Overrides base_color_factor.
    head_face_indices : np.ndarray, optional
        If provided, build two primitives: body (skin color) + head (face texture).

    Returns
    -------
    bytes
        GLB binary data.
    """
    from pygltflib import (
        GLTF2, Scene, Node, Mesh as GltfMesh, Primitive, Attributes, Accessor,
        BufferView, Buffer, Skin, Animation, AnimationChannel,
        AnimationChannelTarget, AnimationSampler,
        Material, PbrMetallicRoughness, TextureInfo,
        Image as GltfImage, Sampler, Texture as GltfTexture,
        UNSIGNED_BYTE, UNSIGNED_SHORT, UNSIGNED_INT, FLOAT,
        VEC2, VEC3, VEC4, MAT4, SCALAR,
        ELEMENT_ARRAY_BUFFER, ARRAY_BUFFER,
        LINEAR, LINEAR_MIPMAP_LINEAR,
    )

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    n_verts = len(vertices)
    n_joints = 24

    with_face = head_face_indices is not None and len(head_face_indices) > 0
    # #region agent log
    import time as _time
    _log_path = r"c:\Users\Syed Taha Hasan\Desktop\Vit\debug-6b80da.log"
    def _dlog(msg, data, hid):
        import json as _j
        with open(_log_path, "a") as _f:
            _f.write(_j.dumps({"sessionId":"6b80da","location":"skinned_glb_builder.py","message":msg,"data":data,"timestamp":int(_time.time()*1000),"hypothesisId":hid})+"\n")
    _dlog("build_skinned_glb entry", {"with_face": with_face, "n_verts": int(n_verts), "base_color_factor": base_color_factor, "head_face_indices_len": int(len(head_face_indices)) if head_face_indices is not None else None}, "A,B")
    # #endregion
    if with_face:
        body_face_mask = np.ones(len(faces), dtype=bool)
        body_face_mask[head_face_indices] = False
        body_face_indices_arr = np.where(body_face_mask)[0]
        body_faces = faces[body_face_indices_arr]
        head_faces = faces[head_face_indices]
        head_verts = np.unique(head_faces.ravel())
        head_center = vertices[head_verts].mean(axis=0).astype(np.float32)
    else:
        body_faces = faces
        head_faces = None
        head_center = None

    # --- Per-vertex: top-4 joint indices and weights (vectorized) ---
    w_arr = np.zeros((n_verts, n_joints), dtype=np.float64)
    n_copy = min(n_verts, len(weights))
    w_arr[:n_copy] = weights[:n_copy]

    top4_indices_arr = np.argsort(w_arr, axis=1)[:, -4:][:, ::-1].astype(np.uint16)
    top4_weights_arr = np.take_along_axis(w_arr, top4_indices_arr.astype(np.int64), axis=1).astype(np.float32)
    w_sums = top4_weights_arr.sum(axis=1, keepdims=True)
    w_sums = np.maximum(w_sums, 1e-8)
    top4_weights_arr /= w_sums

    # --- Compute normals ---
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    # --- Joint hierarchy: compute local translations ---
    joint_local_translation = np.zeros((n_joints, 3), dtype=np.float64)
    for j in range(n_joints):
        parent = int(kintree_parents[j])
        if parent < 0:
            joint_local_translation[j] = joints_positions[j]
        else:
            joint_local_translation[j] = joints_positions[j] - joints_positions[parent]

    # --- Inverse bind matrices ---
    inverse_bind_matrices = np.zeros((n_joints, 4, 4), dtype=np.float32)
    for j in range(n_joints):
        ibm = np.eye(4, dtype=np.float32)
        ibm[0, 3] = -float(joints_positions[j][0])
        ibm[1, 3] = -float(joints_positions[j][1])
        ibm[2, 3] = -float(joints_positions[j][2])
        inverse_bind_matrices[j] = ibm

    # --- Resolve material ---
    mat_info = material_info or {}
    mat_type = mat_info.get("type", "none")

    # === Build binary buffer ===
    buf = io.BytesIO()

    def write_data(data: bytes) -> tuple[int, int]:
        """Write data, return (offset, length). Pad to 4-byte boundary."""
        offset = buf.tell()
        buf.write(data)
        remainder = buf.tell() % 4
        if remainder:
            buf.write(b'\x00' * (4 - remainder))
        return offset, len(data)

    # 0: positions (vec3 float32)
    pos_off, pos_len = write_data(vertices.tobytes())
    # 1: normals (vec3 float32)
    norm_off, norm_len = write_data(normals.tobytes())
    # 2: indices (uint32) — body only when with_face, else all faces
    idx_data = body_faces.flatten().astype(np.uint32).tobytes()
    idx_off, idx_len = write_data(idx_data)
    # 2a: body cylindrical UVs (always generated so wrinkle map can be sampled)
    body_uvs = _generate_body_uvs(vertices)
    body_uv_off, body_uv_len = write_data(body_uvs.tobytes())

    # 2b: head indices (when with_face)
    idx_head_off = idx_head_len = 0
    head_uv_off = head_uv_len = 0
    face_img_off = face_img_len = 0
    if with_face and head_faces is not None:
        idx_head_off, idx_head_len = write_data(head_faces.flatten().astype(np.uint32).tobytes())
        head_uvs = _generate_head_uvs(vertices, faces, head_face_indices, head_center, joints_positions=joints_positions)
        head_uv_off, head_uv_len = write_data(head_uvs.tobytes())
        skin_rgb = (200, 200, 200)
        if base_color_factor and len(base_color_factor) >= 3:
            skin_rgb = (
                int(round(base_color_factor[0] * 255)),
                int(round(base_color_factor[1] * 255)),
                int(round(base_color_factor[2] * 255)),
            )
        face_img_bytes = _generate_face_texture_png(skin_rgb=skin_rgb)
        face_img_off, face_img_len = write_data(face_img_bytes)
    # 3: joints_0 (uvec4 uint16)
    j0_off, j0_len = write_data(top4_indices_arr.tobytes())
    # 4: weights_0 (vec4 float32)
    w0_off, w0_len = write_data(top4_weights_arr.tobytes())
    # 5: inverse bind matrices (mat4 float32)
    # glTF stores matrices in column-major order; numpy .tobytes() is row-major.
    # Transpose each 4x4 so row-major serialisation produces correct column-major layout.
    ibm_col_major = np.ascontiguousarray(inverse_bind_matrices.transpose(0, 2, 1))
    ibm_off, ibm_len = write_data(ibm_col_major.tobytes())

    # Body uses baseColorFactor (same sRGB values as the face texture background),
    # so both primitives render with the exact same skin tone through the same pipeline.

    # Optional: vertex colors (vec4 float32) — only when not using two-primitive face
    vc_off = vc_len = 0
    if not with_face and mat_type == "vertex_colors":
        vc_data = mat_info["colors"].astype(np.float32)
        vc_off, vc_len = write_data(vc_data.tobytes())

    # Optional: UV coordinates (vec2 float32) — for single-primitive texture mesh
    uv_off = uv_len = 0
    if not with_face and mat_type == "texture":
        uv_data = mat_info["uv"].astype(np.float32)
        uv_off, uv_len = write_data(uv_data.tobytes())

    # Optional: texture image (PNG bytes)
    img_off = img_len = 0
    if mat_type == "texture":
        img_off, img_len = write_data(mat_info["png_bytes"])

    # --- Animation data ---
    # Walk animation: 4 keyframes at 0.0, 0.4, 0.8, 1.2 seconds
    walk_times = np.array([0.0, 0.4, 0.8, 1.2], dtype=np.float32)
    walk_time_off, walk_time_len = write_data(walk_times.tobytes())

    walk_kf = _walk_keyframes()
    walk_quat_offsets = {}
    walk_quat_lengths = {}
    for joint_idx, quats in walk_kf.items():
        quat_arr = np.array(quats, dtype=np.float32)
        off, ln = write_data(quat_arr.tobytes())
        walk_quat_offsets[joint_idx] = off
        walk_quat_lengths[joint_idx] = ln

    # Twirl animation: 5 keyframes at 0.0, 0.5, 1.0, 1.5, 2.0 seconds
    twirl_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    twirl_time_off, twirl_time_len = write_data(twirl_times.tobytes())

    twirl_kf = _twirl_keyframes()
    twirl_quat_offsets = {}
    twirl_quat_lengths = {}
    for joint_idx, quats in twirl_kf.items():
        quat_arr = np.array(quats, dtype=np.float32)
        off, ln = write_data(quat_arr.tobytes())
        twirl_quat_offsets[joint_idx] = off
        twirl_quat_lengths[joint_idx] = ln

    # A-pose: 2 keyframes at 0.0, 0.1 seconds (static)
    a_pose_times = np.array([0.0, 0.1], dtype=np.float32)
    a_pose_time_off, a_pose_time_len = write_data(a_pose_times.tobytes())
    a_pose_kf = _a_pose_keyframes()
    a_pose_quat_offsets = {}
    a_pose_quat_lengths = {}
    for joint_idx, quats in a_pose_kf.items():
        quat_arr = np.array(quats, dtype=np.float32)
        off, ln = write_data(quat_arr.tobytes())
        a_pose_quat_offsets[joint_idx] = off
        a_pose_quat_lengths[joint_idx] = ln

    # Natural stand: 2 keyframes at 0.0, 0.1 seconds (static)
    natural_stand_times = np.array([0.0, 0.1], dtype=np.float32)
    natural_stand_time_off, natural_stand_time_len = write_data(natural_stand_times.tobytes())
    natural_stand_kf = _natural_stand_keyframes()
    natural_stand_quat_offsets = {}
    natural_stand_quat_lengths = {}
    for joint_idx, quats in natural_stand_kf.items():
        quat_arr = np.array(quats, dtype=np.float32)
        off, ln = write_data(quat_arr.tobytes())
        natural_stand_quat_offsets[joint_idx] = off
        natural_stand_quat_lengths[joint_idx] = ln

    # T-pose: 2 keyframes (identity rotations for joints we animate elsewhere)
    t_pose_times = np.array([0.0, 0.1], dtype=np.float32)
    t_pose_time_off, t_pose_time_len = write_data(t_pose_times.tobytes())
    t_pose_kf = _t_pose_keyframes()
    t_pose_quat_offsets = {}
    t_pose_quat_lengths = {}
    for joint_idx, quats in t_pose_kf.items():
        quat_arr = np.array(quats, dtype=np.float32)
        off, ln = write_data(quat_arr.tobytes())
        t_pose_quat_offsets[joint_idx] = off
        t_pose_quat_lengths[joint_idx] = ln

    buffer_data = buf.getvalue()

    # === Build glTF structure ===
    gltf = GLTF2()

    # Buffer
    gltf.buffers = [Buffer(byteLength=len(buffer_data))]

    # Buffer views
    buffer_views: list[BufferView] = []
    accessors: list[Accessor] = []

    def add_buffer_view(offset, length, target=None):
        idx = len(buffer_views)
        bv = BufferView(buffer=0, byteOffset=offset, byteLength=length)
        if target is not None:
            bv.target = target
        buffer_views.append(bv)
        return idx

    def add_accessor(bv_idx, component_type, count, accessor_type, min_vals=None, max_vals=None):
        idx = len(accessors)
        acc = Accessor(
            bufferView=bv_idx,
            componentType=component_type,
            count=count,
            type=accessor_type,
        )
        if min_vals is not None:
            acc.min = min_vals
        if max_vals is not None:
            acc.max = max_vals
        accessors.append(acc)
        return idx

    # BV 0: positions
    bv_pos = add_buffer_view(pos_off, pos_len, ARRAY_BUFFER)
    v_min = vertices.min(axis=0).tolist()
    v_max = vertices.max(axis=0).tolist()
    acc_pos = add_accessor(bv_pos, FLOAT, n_verts, VEC3, v_min, v_max)

    # BV 1: normals
    bv_norm = add_buffer_view(norm_off, norm_len, ARRAY_BUFFER)
    acc_norm = add_accessor(bv_norm, FLOAT, n_verts, VEC3)

    # BV 2: indices (body primitive)
    bv_idx = add_buffer_view(idx_off, idx_len, ELEMENT_ARRAY_BUFFER)
    n_body_indices = body_faces.size
    acc_idx = add_accessor(bv_idx, UNSIGNED_INT, n_body_indices, SCALAR)

    # BV 2a: body cylindrical UVs
    bv_body_uv = add_buffer_view(body_uv_off, body_uv_len, ARRAY_BUFFER)
    acc_body_uv = add_accessor(bv_body_uv, FLOAT, n_verts, VEC2)

    # BV 2b: head indices (when with_face)
    bv_idx_head = None
    acc_idx_head = None
    if with_face and idx_head_len > 0:
        bv_idx_head = add_buffer_view(idx_head_off, idx_head_len, ELEMENT_ARRAY_BUFFER)
        n_head_indices = head_faces.size
        acc_idx_head = add_accessor(bv_idx_head, UNSIGNED_INT, n_head_indices, SCALAR)

    # Head UV and face texture (when with_face)
    acc_head_uv = None
    bv_face_img = None
    if with_face and head_uv_len > 0:
        bv_head_uv = add_buffer_view(head_uv_off, head_uv_len, ARRAY_BUFFER)
        acc_head_uv = add_accessor(bv_head_uv, FLOAT, n_verts, VEC2)
    if with_face and face_img_len > 0:
        bv_face_img = add_buffer_view(face_img_off, face_img_len)

    # BV 3: joints_0
    bv_j0 = add_buffer_view(j0_off, j0_len, ARRAY_BUFFER)
    acc_j0 = add_accessor(bv_j0, UNSIGNED_SHORT, n_verts, VEC4)

    # BV 4: weights_0
    bv_w0 = add_buffer_view(w0_off, w0_len, ARRAY_BUFFER)
    acc_w0 = add_accessor(bv_w0, FLOAT, n_verts, VEC4)

    # BV 5: inverse bind matrices
    bv_ibm = add_buffer_view(ibm_off, ibm_len)
    acc_ibm = add_accessor(bv_ibm, FLOAT, n_joints, MAT4)

    # Body uses baseColorFactor — no vertex color accessor needed.

    # Optional: vertex colors accessor (single-primitive mesh)
    acc_vc = None
    if not with_face and mat_type == "vertex_colors" and vc_len > 0:
        bv_vc = add_buffer_view(vc_off, vc_len, ARRAY_BUFFER)
        acc_vc = add_accessor(bv_vc, FLOAT, n_verts, VEC4)

    # Optional: UV accessor (single-primitive texture mesh)
    acc_uv = None
    if not with_face and mat_type == "texture" and uv_len > 0:
        bv_uv = add_buffer_view(uv_off, uv_len, ARRAY_BUFFER)
        acc_uv = add_accessor(bv_uv, FLOAT, n_verts, VEC2)

    # --- Animation buffer views and accessors ---
    # Walk time
    bv_walk_time = add_buffer_view(walk_time_off, walk_time_len)
    acc_walk_time = add_accessor(bv_walk_time, FLOAT, len(walk_times), SCALAR,
                                  [float(walk_times[0])], [float(walk_times[-1])])

    walk_acc_quats = {}
    for joint_idx in walk_kf:
        bv = add_buffer_view(walk_quat_offsets[joint_idx], walk_quat_lengths[joint_idx])
        acc = add_accessor(bv, FLOAT, len(walk_kf[joint_idx]), VEC4)
        walk_acc_quats[joint_idx] = acc

    # Twirl time
    bv_twirl_time = add_buffer_view(twirl_time_off, twirl_time_len)
    acc_twirl_time = add_accessor(bv_twirl_time, FLOAT, len(twirl_times), SCALAR,
                                   [float(twirl_times[0])], [float(twirl_times[-1])])

    twirl_acc_quats = {}
    for joint_idx in twirl_kf:
        bv = add_buffer_view(twirl_quat_offsets[joint_idx], twirl_quat_lengths[joint_idx])
        acc = add_accessor(bv, FLOAT, len(twirl_kf[joint_idx]), VEC4)
        twirl_acc_quats[joint_idx] = acc

    # A-pose time + quats
    bv_a_pose_time = add_buffer_view(a_pose_time_off, a_pose_time_len)
    acc_a_pose_time = add_accessor(bv_a_pose_time, FLOAT, len(a_pose_times), SCALAR,
                                    [float(a_pose_times[0])], [float(a_pose_times[-1])])
    a_pose_acc_quats = {}
    for joint_idx in a_pose_kf:
        bv = add_buffer_view(a_pose_quat_offsets[joint_idx], a_pose_quat_lengths[joint_idx])
        acc = add_accessor(bv, FLOAT, len(a_pose_kf[joint_idx]), VEC4)
        a_pose_acc_quats[joint_idx] = acc

    # Natural stand time + quats
    bv_natural_stand_time = add_buffer_view(natural_stand_time_off, natural_stand_time_len)
    acc_natural_stand_time = add_accessor(bv_natural_stand_time, FLOAT, len(natural_stand_times), SCALAR,
                                          [float(natural_stand_times[0])], [float(natural_stand_times[-1])])
    natural_stand_acc_quats = {}
    for joint_idx in natural_stand_kf:
        bv = add_buffer_view(natural_stand_quat_offsets[joint_idx], natural_stand_quat_lengths[joint_idx])
        acc = add_accessor(bv, FLOAT, len(natural_stand_kf[joint_idx]), VEC4)
        natural_stand_acc_quats[joint_idx] = acc

    # T-pose time + quats
    bv_t_pose_time = add_buffer_view(t_pose_time_off, t_pose_time_len)
    acc_t_pose_time = add_accessor(bv_t_pose_time, FLOAT, len(t_pose_times), SCALAR,
                                   [float(t_pose_times[0])], [float(t_pose_times[-1])])
    t_pose_acc_quats = {}
    for joint_idx in t_pose_kf:
        bv = add_buffer_view(t_pose_quat_offsets[joint_idx], t_pose_quat_lengths[joint_idx])
        acc = add_accessor(bv, FLOAT, len(t_pose_kf[joint_idx]), VEC4)
        t_pose_acc_quats[joint_idx] = acc

    gltf.bufferViews = buffer_views
    gltf.accessors = accessors

    # === Material ===
    gltf.images = []
    gltf.samplers = []
    gltf.textures = []

    if with_face:
        # Both body and face primitives use the same sRGB skin color via baseColorFactor.
        raw = base_color_factor or _SKIN_COLOR_FACTOR
        color = [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]) if len(raw) >= 4 else 1.0]
        logger.info("Building skinned GLB: with_face=%s, base_color_factor=%s", with_face, base_color_factor)
        mat_body = Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=color,
                metallicFactor=0.0,
                roughnessFactor=0.9,
            ),
            doubleSided=True,
        )
        if bv_face_img is not None:
            gltf.images = [GltfImage(bufferView=bv_face_img, mimeType="image/png")]
            gltf.samplers = [Sampler(magFilter=LINEAR, minFilter=LINEAR_MIPMAP_LINEAR)]
            gltf.textures = [GltfTexture(source=0, sampler=0)]
        mat_face = Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorTexture=TextureInfo(index=0) if bv_face_img is not None else None,
                baseColorFactor=color if bv_face_img is None else [1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.9,
            ),
            doubleSided=True,
        )
        gltf.materials = [mat_body, mat_face]
    elif mat_type == "texture" and acc_uv is not None:
        # Texture material: embed image, create sampler+texture
        bv_img = add_buffer_view(img_off, img_len)  # no target for images
        gltf.bufferViews = buffer_views  # refresh after adding image BV
        gltf.images = [GltfImage(bufferView=bv_img, mimeType="image/png")]
        gltf.samplers = [Sampler(magFilter=LINEAR, minFilter=LINEAR_MIPMAP_LINEAR)]
        gltf.textures = [GltfTexture(source=0, sampler=0)]
        mat = Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorTexture=TextureInfo(index=0),
                metallicFactor=0.0,
                roughnessFactor=0.9,
            ),
            doubleSided=True,
        )
        gltf.materials = [mat]
    elif mat_type == "vertex_colors":
        # Vertex color material — let vertex colors drive appearance
        mat = Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],  # white so vertex colors show
                metallicFactor=0.0,
                roughnessFactor=0.9,
            ),
            doubleSided=True,
        )
        gltf.materials = [mat]
    elif mat_type == "color":
        # Solid color from material_info
        mat = Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=mat_info["base_color_factor"],
                metallicFactor=0.0,
                roughnessFactor=0.9,
            ),
            doubleSided=True,
        )
        gltf.materials = [mat]
    else:
        raw = base_color_factor or _SKIN_COLOR_FACTOR
        color = [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]) if len(raw) >= 4 else 1.0]
        logger.info("Building skinned GLB (default path): mat_type=%s, resolved_color=%s", mat_type, color)
        mat = Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=color,
                metallicFactor=0.0,
                roughnessFactor=0.9,
            ),
            doubleSided=True,
        )
        gltf.materials = [mat]

    # === Nodes: skeleton joints + mesh node ===
    nodes = []
    for j in range(n_joints):
        t = joint_local_translation[j]
        node = Node(
            name=_JOINT_NAMES[j],
            translation=[float(t[0]), float(t[1]), float(t[2])],
        )
        children = [c for c in range(n_joints) if int(kintree_parents[c]) == j]
        if children:
            node.children = children
        nodes.append(node)

    # Node 24: mesh node (skinned)
    mesh_node_idx = n_joints
    mesh_node = Node(name="body", mesh=0, skin=0)
    nodes.append(mesh_node)

    # Node 25: root scene node containing skeleton root + mesh
    root_node_idx = n_joints + 1
    root_node = Node(name="root", children=[0, mesh_node_idx])
    nodes.append(root_node)

    gltf.nodes = nodes

    # === Skin ===
    joint_indices = list(range(n_joints))
    skin = Skin(
        name="SMPL_skeleton",
        inverseBindMatrices=acc_ibm,
        skeleton=0,
        joints=joint_indices,
    )
    gltf.skins = [skin]

    # === Mesh ===
    if with_face and acc_idx_head is not None and acc_head_uv is not None:
        # Two primitives: body (material 0, cylindrical UVs) + head (material 1, face texture)
        attrs_body = Attributes()
        attrs_body.POSITION = acc_pos
        attrs_body.NORMAL = acc_norm
        attrs_body.JOINTS_0 = acc_j0
        attrs_body.WEIGHTS_0 = acc_w0
        attrs_body.TEXCOORD_0 = acc_body_uv
        prim_body = Primitive(attributes=attrs_body, indices=acc_idx, material=0)
        attrs_head = Attributes()
        attrs_head.POSITION = acc_pos
        attrs_head.NORMAL = acc_norm
        attrs_head.JOINTS_0 = acc_j0
        attrs_head.WEIGHTS_0 = acc_w0
        attrs_head.TEXCOORD_0 = acc_head_uv
        prim_head = Primitive(attributes=attrs_head, indices=acc_idx_head, material=1)
        gltf_mesh = GltfMesh(name="body_mesh", primitives=[prim_body, prim_head])
    else:
        attrs = Attributes()
        attrs.POSITION = acc_pos
        attrs.NORMAL = acc_norm
        attrs.JOINTS_0 = acc_j0
        attrs.WEIGHTS_0 = acc_w0
        if acc_vc is not None:
            attrs.COLOR_0 = acc_vc
        if acc_uv is not None:
            attrs.TEXCOORD_0 = acc_uv
        primitive = Primitive(attributes=attrs, indices=acc_idx, material=0)
        gltf_mesh = GltfMesh(name="body_mesh", primitives=[primitive])
    gltf.meshes = [gltf_mesh]

    # === Animations ===
    animations = []

    # Walk animation
    walk_samplers = []
    walk_channels = []
    for joint_idx in sorted(walk_kf.keys()):
        sampler_idx = len(walk_samplers)
        walk_samplers.append(AnimationSampler(
            input=acc_walk_time,
            output=walk_acc_quats[joint_idx],
            interpolation="LINEAR",
        ))
        walk_channels.append(AnimationChannel(
            sampler=sampler_idx,
            target=AnimationChannelTarget(node=joint_idx, path="rotation"),
        ))
    animations.append(Animation(name="walk", samplers=walk_samplers, channels=walk_channels))

    # Twirl animation
    twirl_samplers = []
    twirl_channels = []
    for joint_idx in sorted(twirl_kf.keys()):
        sampler_idx = len(twirl_samplers)
        twirl_samplers.append(AnimationSampler(
            input=acc_twirl_time,
            output=twirl_acc_quats[joint_idx],
            interpolation="LINEAR",
        ))
        twirl_channels.append(AnimationChannel(
            sampler=sampler_idx,
            target=AnimationChannelTarget(node=joint_idx, path="rotation"),
        ))
    animations.append(Animation(name="twirl", samplers=twirl_samplers, channels=twirl_channels))

    # A-pose animation
    a_pose_samplers = []
    a_pose_channels = []
    for joint_idx in sorted(a_pose_kf.keys()):
        sampler_idx = len(a_pose_samplers)
        a_pose_samplers.append(AnimationSampler(
            input=acc_a_pose_time,
            output=a_pose_acc_quats[joint_idx],
            interpolation="LINEAR",
        ))
        a_pose_channels.append(AnimationChannel(
            sampler=sampler_idx,
            target=AnimationChannelTarget(node=joint_idx, path="rotation"),
        ))
    animations.append(Animation(name="a_pose", samplers=a_pose_samplers, channels=a_pose_channels))

    # Natural stand animation
    natural_stand_samplers = []
    natural_stand_channels = []
    for joint_idx in sorted(natural_stand_kf.keys()):
        sampler_idx = len(natural_stand_samplers)
        natural_stand_samplers.append(AnimationSampler(
            input=acc_natural_stand_time,
            output=natural_stand_acc_quats[joint_idx],
            interpolation="LINEAR",
        ))
        natural_stand_channels.append(AnimationChannel(
            sampler=sampler_idx,
            target=AnimationChannelTarget(node=joint_idx, path="rotation"),
        ))
    animations.append(Animation(name="natural_stand", samplers=natural_stand_samplers, channels=natural_stand_channels))

    # T-pose animation (identity rotations so selecting T-Pose resets skeleton)
    t_pose_samplers = []
    t_pose_channels = []
    for joint_idx in sorted(t_pose_kf.keys()):
        sampler_idx = len(t_pose_samplers)
        t_pose_samplers.append(AnimationSampler(
            input=acc_t_pose_time,
            output=t_pose_acc_quats[joint_idx],
            interpolation="LINEAR",
        ))
        t_pose_channels.append(AnimationChannel(
            sampler=sampler_idx,
            target=AnimationChannelTarget(node=joint_idx, path="rotation"),
        ))
    animations.append(Animation(name="t_pose", samplers=t_pose_samplers, channels=t_pose_channels))

    gltf.animations = animations

    # === Scene ===
    gltf.scenes = [Scene(nodes=[root_node_idx])]
    gltf.scene = 0

    # === Set binary blob ===
    gltf.set_binary_blob(buffer_data)

    # === Export to bytes ===
    return b"".join(gltf.save_to_bytes())


def build_skinned_garment_glb(
    garment_glb_bytes: bytes,
    body_weights: np.ndarray,
    selected_vertex_indices: np.ndarray,
    kintree_parents: np.ndarray,
    joints_positions: np.ndarray,
) -> bytes:
    """Build a skinned+animated GLB for a conforming garment.

    The garment's skinning weights are transferred from the body mesh:
    garment_weights[i] = body_weights[selected_vertex_indices[i]]
    Preserves the garment's color/texture from the source GLB.

    Parameters
    ----------
    garment_glb_bytes : bytes
        Raw GLB bytes of the garment mesh (from process_garment).
    body_weights : np.ndarray, shape (6890, 24)
        Body skinning weights.
    selected_vertex_indices : np.ndarray
        Maps each garment vertex to a body vertex index.
    kintree_parents : np.ndarray, shape (24,)
        Parent joint indices.
    joints_positions : np.ndarray, shape (24, 3)
        Rest-pose joint positions.

    Returns
    -------
    bytes
        Skinned+animated GLB bytes.
    """
    # Load garment mesh from GLB bytes (preserving visual data)
    garment_scene = trimesh.load(io.BytesIO(garment_glb_bytes), file_type="glb")
    if isinstance(garment_scene, trimesh.Scene):
        geoms = [g for g in garment_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            logger.warning("No geometry in garment GLB, returning original")
            return garment_glb_bytes
        garment_mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    elif isinstance(garment_scene, trimesh.Trimesh):
        garment_mesh = garment_scene
    else:
        logger.warning("Garment GLB did not load as Trimesh, returning original")
        return garment_glb_bytes

    # Trimesh loses embedded texture images on GLB reload: PBRMaterial.baseColorTexture
    # becomes None and baseColorFactor becomes grey [102,102,102,255]. So we always
    # try to extract the embedded image directly from the raw GLB bytes first.
    mat_info: dict = {"type": "none"}
    visual = garment_mesh.visual
    uv = None
    if isinstance(visual, trimesh.visual.TextureVisuals) and visual.uv is not None:
        uv = np.asarray(visual.uv, dtype=np.float32)
        if len(uv) != len(garment_mesh.vertices):
            uv = None
    if uv is not None:
        img_bytes = _extract_first_image_from_glb(garment_glb_bytes)
        if img_bytes:
            mat_info = {"type": "texture", "png_bytes": img_bytes, "uv": uv}
            logger.info("Extracted garment texture (%d bytes) from GLB for skinned mesh", len(img_bytes))
    if mat_info["type"] == "none":
        mat_info = _extract_material_from_mesh(garment_mesh)

    # Transfer weights: garment vertex i -> body vertex selected_vertex_indices[i]
    n_garment_verts = len(garment_mesh.vertices)
    sel = np.asarray(selected_vertex_indices, dtype=np.int64)
    garment_weights = np.zeros((n_garment_verts, 24), dtype=np.float64)
    n_mapped = min(n_garment_verts, len(sel))
    valid = sel[:n_mapped] < len(body_weights)
    garment_weights[:n_mapped][valid] = body_weights[sel[:n_mapped][valid]]
    # Default unmapped vertices to root joint
    unmapped = ~valid
    if np.any(unmapped):
        garment_weights[:n_mapped][unmapped, 0] = 1.0
    if n_mapped < n_garment_verts:
        garment_weights[n_mapped:, 0] = 1.0

    return build_skinned_glb(
        garment_mesh,
        garment_weights,
        kintree_parents,
        joints_positions,
        material_info=mat_info,
    )
