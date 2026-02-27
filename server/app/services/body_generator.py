"""Body Generator Service – v2.

Builds a parametric 3D human body from body measurements.  Each major
body region (trunk, arms, legs) is constructed as a **continuous lofted
tube** with many cross-sections so that the silhouette looks smooth and
organic rather than a collection of disconnected cylinders.

When *use_base_mesh* is True, loads a pre-made OBJ base mesh (e.g. FinalBaseMesh.obj),
scales it to the requested height, and exports as GLB.
Coordinate convention: Y-up, right-handed; feet near y = 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from ..models.body import BodyMeasurements, Gender
from ..utils.mesh_helpers import (
    apply_face_color,
    create_lofted_tube,
)

# Path to optional OBJ base mesh (male body); used when use_base_mesh=True
_BASE_MESH_PATH = Path(__file__).resolve().parent.parent / "assets" / "FinalBaseMesh.obj"

# ---------------------------------------------------------------------------
# Gender-specific proportion constants (fractions of total height)
# ---------------------------------------------------------------------------
_PROPORTIONS: dict[Gender, dict[str, float]] = {
    Gender.MALE: {
        "head_ratio": 0.120,
        "neck_ratio": 0.030,
        "torso_depth_factor": 0.85,
        "shoulder_drop": 0.020,
        "arm_radius_factor": 0.040,
        "leg_radius_factor": 0.058,
        "hand_radius_factor": 0.025,
        "foot_length_factor": 0.150,
        "foot_height_factor": 0.020,
    },
    Gender.FEMALE: {
        "head_ratio": 0.115,
        "neck_ratio": 0.028,
        "torso_depth_factor": 0.80,
        "shoulder_drop": 0.025,
        "arm_radius_factor": 0.036,
        "leg_radius_factor": 0.052,
        "hand_radius_factor": 0.022,
        "foot_length_factor": 0.140,
        "foot_height_factor": 0.018,
    },
}

# Warm-beige mannequin skin tone (RGBA)
_SKIN_COLOR = (222, 195, 170, 255)

# Mesh resolution
_RING_POINTS = 48      # vertices per cross-section ring
_TRUNK_SECTIONS = 40   # cross-sections for trunk
_LIMB_SECTIONS = 20    # cross-sections per limb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skin_color_factor(measurements: BodyMeasurements) -> list[float] | None:
    """Return [r, g, b, 1] from measurements.skin_color_hex if set, else None."""
    hex_str = getattr(measurements, "skin_color_hex", None)
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _get_body_mesh_from_base_mesh(measurements: BodyMeasurements, mesh_path: Path) -> trimesh.Trimesh:
    """Load an OBJ base mesh, scale to height, center (feet at y=0), apply skin color. Returns mesh."""
    scene = trimesh.load(mesh_path, force="mesh")
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])
    else:
        mesh = scene
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Base mesh did not load as a single mesh")

    bounds = mesh.bounds
    y_min, y_max = float(bounds[0][1]), float(bounds[1][1])
    mesh_height = y_max - y_min
    if mesh_height <= 0:
        mesh_height = 1.0
    target_height_m = measurements.height_cm / 100.0
    scale = target_height_m / mesh_height
    mesh.apply_scale(scale)
    mesh.apply_translation([-mesh.centroid[0], -mesh.bounds[0][1], -mesh.centroid[2]])
    apply_face_color(mesh, _SKIN_COLOR)
    trimesh.repair.fix_normals(mesh)
    _ = mesh.vertex_normals
    return mesh


def get_body_mesh(measurements: BodyMeasurements) -> tuple[trimesh.Trimesh, dict | None, dict | None]:
    """
    Build the body mesh from measurements. Returns (mesh, landmarks, skinning_data).
    skinning_data is a dict with 'weights', 'kintree_parents', 'joints_positions'
    when SMPL is used, or None for parametric/base mesh bodies.
    """
    if getattr(measurements, "use_smpl", False):
        try:
            from .smpl_body import generate_body_smpl
            mesh, landmarks, skinning_data = generate_body_smpl(measurements)
            return mesh, landmarks, skinning_data
        except FileNotFoundError:
            # SMPL files missing or unconverted — re-raise so user knows
            raise
        except Exception as exc:
            import logging
            log = logging.getLogger(__name__)
            log.warning(
                "SMPL body generation failed, falling back to parametric body: %s: %s",
                type(exc).__name__, exc,
            )
            # Fall through to parametric or base mesh below
    if getattr(measurements, "use_base_mesh", False) and _BASE_MESH_PATH.is_file():
        return _get_body_mesh_from_base_mesh(measurements, _BASE_MESH_PATH), None, None
    # Parametric path: build mesh (same logic as generate_body below)
    mesh, _ = _build_parametric_body(measurements)
    return mesh, None, None


def generate_body(measurements: BodyMeasurements) -> tuple[bytes, dict | None, str]:
    """Generate a GLB body mesh from *measurements*.

    When SMPL is used, exports a skinned+animated GLB.
    Returns (glb_bytes, landmarks, model_type).
    """
    import logging
    log = logging.getLogger(__name__)

    # #region agent log
    import json as _j, time as _t
    _lp = r"c:\Users\Syed Taha Hasan\Desktop\Vit\debug-6b80da.log"
    def _dlog2(msg, data, hid):
        try:
            with open(_lp, "a") as _f:
                _f.write(_j.dumps({"sessionId":"6b80da","location":"body_generator.py","message":msg,"data":data,"timestamp":int(_t.time()*1000),"hypothesisId":hid})+"\n")
        except Exception as _e:
            log.error("_dlog2 failed: %s", _e)
    _dlog2("generate_body called", {"use_smpl": getattr(measurements, "use_smpl", None), "use_base_mesh": getattr(measurements, "use_base_mesh", None), "skin_color_hex": getattr(measurements, "skin_color_hex", None)}, "PATH")
    # #endregion

    mesh, landmarks, skinning_data = get_body_mesh(measurements)
    # #region agent log
    _dlog2("get_body_mesh returned", {"skinning_data_is_none": skinning_data is None, "has_landmarks": landmarks is not None}, "PATH")
    # #endregion
    if skinning_data is not None:
        from .skinned_glb_builder import build_skinned_glb
        base_color = _skin_color_factor(measurements)
        head_fi = skinning_data.get("head_face_indices")
        log.info(
            "generate_body: skin_color_hex=%r, base_color=%s, head_face_indices=%s",
            getattr(measurements, "skin_color_hex", None),
            base_color,
            f"({len(head_fi)} faces)" if head_fi is not None else None,
        )
        # #region agent log
        _dlog2("calling build_skinned_glb", {"base_color": base_color, "head_fi_len": len(head_fi) if head_fi is not None else None}, "PATH")
        # #endregion
        glb_data = build_skinned_glb(
            mesh,
            skinning_data["weights"],
            skinning_data["kintree_parents"],
            skinning_data["joints_positions"],
            head_face_indices=head_fi,
            base_color_factor=base_color,
        )
        # #region agent log
        _dlog2("build_skinned_glb returned", {"glb_size": len(glb_data)}, "PATH")
        # #endregion
        return glb_data, landmarks, "smpl"
    else:
        # #region agent log
        _dlog2("using non-SMPL path (mesh.export)", {"use_base_mesh": getattr(measurements, "use_base_mesh", None)}, "PATH")
        # #endregion
        glb_data = mesh.export(file_type="glb")
        model_type = "base_mesh" if getattr(measurements, "use_base_mesh", False) else "parametric"
        return glb_data, landmarks, model_type


def _build_parametric_body(measurements: BodyMeasurements) -> tuple[trimesh.Trimesh, None]:
    """Build parametric body mesh. Returns (mesh, None)."""
    p = _PROPORTIONS[measurements.gender]

    # Convert to metres
    h = measurements.height_cm / 100.0
    chest_c = measurements.chest_cm / 100.0
    waist_c = measurements.waist_cm / 100.0
    hip_c = measurements.hip_cm / 100.0
    shoulder_w = measurements.shoulder_width_cm / 100.0
    arm_len = measurements.arm_length_cm / 100.0
    inseam = measurements.inseam_cm / 100.0

    # Elliptical radii from circumferences
    df = p["torso_depth_factor"]
    chest_rx = chest_c / (2.0 * np.pi)
    chest_rz = chest_rx * df
    waist_rx = waist_c / (2.0 * np.pi)
    waist_rz = waist_rx * df
    hip_rx = hip_c / (2.0 * np.pi)
    hip_rz = hip_rx * df

    # Vertical segment heights
    head_h = h * p["head_ratio"]
    neck_h = h * p["neck_ratio"]
    torso_h = h - inseam - head_h - neck_h
    upper_leg_h = inseam * 0.54
    lower_leg_h = inseam * 0.46
    upper_arm_h = arm_len * 0.48
    lower_arm_h = arm_len * 0.52

    # Key Y positions
    torso_base_y = inseam
    torso_top_y = inseam + torso_h
    neck_top_y = torso_top_y + neck_h
    head_centre_y = neck_top_y + head_h * 0.50
    shoulder_y = torso_top_y - h * p["shoulder_drop"]

    # Derived radii
    neck_r = head_h / 2.0 * 0.45
    arm_r = h * p["arm_radius_factor"]
    leg_r = h * p["leg_radius_factor"]
    leg_gap = hip_rx * 0.55
    hand_r = h * p["hand_radius_factor"]
    foot_len = h * p["foot_length_factor"]
    foot_h = h * p["foot_height_factor"]

    # -------------------------------------------------------------------
    # Build body parts
    # -------------------------------------------------------------------
    parts: list[trimesh.Trimesh] = []

    # 1) HEAD – slightly elongated ellipsoid
    head = _create_head(head_h / 2.0)
    head.apply_translation([0.0, head_centre_y, 0.0])
    parts.append(head)

    # 2) TRUNK – continuous loft from pelvis through torso and neck
    trunk = _create_trunk(
        chest_rx, chest_rz,
        waist_rx, waist_rz,
        hip_rx, hip_rz,
        shoulder_w / 2.0,
        neck_r,
        torso_h, neck_h,
    )
    trunk.apply_translation([0.0, torso_base_y, 0.0])
    parts.append(trunk)

    # 3) ARMS
    for side in (-1.0, 1.0):
        arm = _create_arm(arm_r, upper_arm_h, lower_arm_h)
        arm.apply_translation([side * shoulder_w / 2.0, shoulder_y, 0.0])
        parts.append(arm)

        # Hand
        hand_mesh = trimesh.creation.icosphere(subdivisions=2, radius=hand_r)
        hand_y = shoulder_y - upper_arm_h - lower_arm_h - hand_r * 0.6
        hand_mesh.apply_translation([side * shoulder_w / 2.0, hand_y, 0.0])
        parts.append(hand_mesh)

    # 4) LEGS
    for side in (-1.0, 1.0):
        leg = _create_leg(leg_r, upper_leg_h, lower_leg_h)
        leg.apply_translation([side * leg_gap, 0.0, 0.0])
        parts.append(leg)

        # Foot – elongated ellipsoid (rounded, natural shape)
        foot_w = leg_r * 1.2
        foot_mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        foot_mesh.vertices *= np.array([foot_w, foot_h, foot_len])
        foot_mesh.apply_translation([side * leg_gap, foot_h / 2.0, foot_len * 0.15])
        parts.append(foot_mesh)

    # -------------------------------------------------------------------
    # Assemble, colour, fix normals, export
    # -------------------------------------------------------------------
    body = trimesh.util.concatenate(parts)
    apply_face_color(body, _SKIN_COLOR)
    trimesh.repair.fix_normals(body)

    _ = body.vertex_normals  # noqa: F841
    return body, None


# ---------------------------------------------------------------------------
# Part builders
# ---------------------------------------------------------------------------

def _create_head(radius: float) -> trimesh.Trimesh:
    """Slightly elongated ellipsoid (taller than wide)."""
    head = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    head.vertices[:, 1] *= 1.15
    return head


def _create_trunk(
    chest_rx: float, chest_rz: float,
    waist_rx: float, waist_rz: float,
    hip_rx: float, hip_rz: float,
    shoulder_half_w: float,
    neck_r: float,
    torso_h: float, neck_h: float,
) -> trimesh.Trimesh:
    """Create a continuous lofted tube from pelvis (y=0) to top of neck."""
    total_h = torso_h + neck_h

    torso_t = torso_h / total_h
    waist_t = torso_t * 0.38

    key_rx = [
        (0.00,       hip_rx),
        (0.06,       hip_rx * 1.01),
        (0.15,       hip_rx * 0.97),
        (waist_t,    waist_rx),
        (torso_t * 0.60, (waist_rx + chest_rx) * 0.52),
        (torso_t * 0.78, chest_rx),
        (torso_t * 0.90, chest_rx * 0.95),
        (torso_t * 0.94, shoulder_half_w * 0.92),
        (torso_t * 0.97, shoulder_half_w * 0.65),
        (torso_t,        neck_r * 1.45),
        (torso_t + (1.0 - torso_t) * 0.3, neck_r * 1.10),
        (1.00,           neck_r),
    ]

    key_rz = [
        (0.00,       hip_rz),
        (0.06,       hip_rz * 1.01),
        (0.15,       hip_rz * 0.95),
        (waist_t,    waist_rz),
        (torso_t * 0.60, (waist_rz + chest_rz) * 0.52),
        (torso_t * 0.78, chest_rz),
        (torso_t * 0.90, chest_rz * 0.94),
        (torso_t * 0.94, neck_r * 1.20),
        (torso_t * 0.97, neck_r * 1.15),
        (torso_t,        neck_r * 1.30),
        (torso_t + (1.0 - torso_t) * 0.3, neck_r * 1.00),
        (1.00,           neck_r * 0.95),
    ]

    ts_key_rx = np.array([kp[0] for kp in key_rx])
    vs_key_rx = np.array([kp[1] for kp in key_rx])
    ts_key_rz = np.array([kp[0] for kp in key_rz])
    vs_key_rz = np.array([kp[1] for kp in key_rz])

    n = _TRUNK_SECTIONS
    t_samples = np.linspace(0.0, 1.0, n)
    rx = np.interp(t_samples, ts_key_rx, vs_key_rx)
    rz = np.interp(t_samples, ts_key_rz, vs_key_rz)
    heights = t_samples * total_h

    return create_lofted_tube(rx, rz, heights, ring_points=_RING_POINTS)


def _create_arm(base_r: float, upper_h: float, lower_h: float) -> trimesh.Trimesh:
    """Create an arm as a lofted tube hanging downward from y=0."""
    total_h = upper_h + lower_h

    key = [
        (0.00, base_r * 1.15),
        (0.08, base_r * 0.95),
        (0.22, base_r * 0.90),
        (0.35, base_r * 0.82),
        (0.48, base_r * 0.74),
        (0.55, base_r * 0.72),
        (0.70, base_r * 0.67),
        (0.85, base_r * 0.60),
        (1.00, base_r * 0.52),
    ]

    ts = np.array([k[0] for k in key])
    vs = np.array([k[1] for k in key])

    n = _LIMB_SECTIONS
    t_samples = np.linspace(0.0, 1.0, n)
    r_vals = np.interp(t_samples, ts, vs)

    heights = -t_samples * total_h
    rx = r_vals
    rz = r_vals * 0.90

    return create_lofted_tube(rx, rz, heights, ring_points=_RING_POINTS)


def _create_leg(base_r: float, upper_h: float, lower_h: float) -> trimesh.Trimesh:
    """Create a leg from floor (y=0) to hip (y = upper_h + lower_h)."""
    total_h = upper_h + lower_h

    key = [
        (0.00, base_r * 0.48),
        (0.08, base_r * 0.52),
        (0.20, base_r * 0.62),
        (0.30, base_r * 0.58),
        (0.38, base_r * 0.55),
        (0.44, base_r * 0.60),
        (0.55, base_r * 0.72),
        (0.70, base_r * 0.84),
        (0.85, base_r * 0.94),
        (1.00, base_r * 1.00),
    ]

    ts = np.array([k[0] for k in key])
    vs = np.array([k[1] for k in key])

    n = _LIMB_SECTIONS + 4
    t_samples = np.linspace(0.0, 1.0, n)
    r_vals = np.interp(t_samples, ts, vs)

    heights = t_samples * total_h
    rx = r_vals
    rz = r_vals * 0.88

    return create_lofted_tube(rx, rz, heights, ring_points=_RING_POINTS)
