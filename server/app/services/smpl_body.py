"""
SMPL body generation — loads pre-converted pkl files directly.

The pkl files must be converted to numpy-only format first
(run ``python convert_smpl.py`` once after downloading SMPL 1.1.0).

Generates a mesh-only GLB in T-pose, scaled to the user's height.
Also returns body landmarks so that garments can be positioned correctly,
and skinning data (weights, kintree, joint positions) for animation.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np
import trimesh

from ..models.body import BodyMeasurements, Gender
from ..utils.mesh_helpers import apply_face_color

_SKIN_COLOR = (222, 195, 170, 255)

# SMPL model folder; must contain SMPL_MALE.pkl / SMPL_FEMALE.pkl (numpy-only)
_SMPL_MODEL_DIR = Path(__file__).resolve().parent.parent / "assets" / "smpl"

# SMPL joint names (standard 24-joint skeleton)
_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]


class BodyLandmarks(TypedDict):
    """Key body positioning data for garment fitting."""
    shoulder_y: float
    shoulder_half_w: float
    waist_y: float
    pelvis_y: float
    hip_rx: float
    neck_y: float
    inseam_y: float       # crotch height (= ankle_y + leg length from ankle)
    ankle_y: float
    arm_r: float
    chest_rx: float
    chest_rz: float
    waist_rx: float
    waist_rz: float


class SkinningData(TypedDict):
    """Skinning data needed for animated GLB export."""
    weights: np.ndarray            # (6890, 24) float64 — skinning weights
    kintree_parents: np.ndarray    # (24,) int — parent joint index for each joint
    joints_positions: np.ndarray   # (24, 3) float64 — rest-pose joint positions


def _smpl_path_for_gender(gender: Gender) -> Path | None:
    """Return path to SMPL pkl for gender, or None if not found."""
    if not _SMPL_MODEL_DIR.is_dir():
        return None
    name = "SMPL_MALE.pkl" if gender == Gender.MALE else "SMPL_FEMALE.pkl"
    path = _SMPL_MODEL_DIR / name
    return path if path.is_file() else None


def _lbs_forward(
    v_template: np.ndarray,
    shapedirs: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """
    Run SMPL linear blend shapes (shape only, no pose blend shapes).

    Returns vertices (6890, 3).
    """
    n_betas = min(betas.shape[0], shapedirs.shape[2])
    return v_template + np.einsum("vci,i->vc", shapedirs[:, :, :n_betas], betas[:n_betas])


def _estimate_cross_section_radii(
    vertices: np.ndarray, y_target: float, tolerance: float = 0.02
) -> tuple[float, float]:
    """Estimate (rx, rz) at a horizontal slice of the body mesh."""
    mask = np.abs(vertices[:, 1] - y_target) < tolerance
    if mask.sum() < 10:
        mask = np.abs(vertices[:, 1] - y_target) < tolerance * 3
    if mask.sum() < 3:
        return (0.1, 0.1)
    ring = vertices[mask]
    # Only consider torso (exclude arms): X within ±0.25 m from centre
    torso_mask = np.abs(ring[:, 0]) < 0.25
    if torso_mask.sum() >= 5:
        ring = ring[torso_mask]
    rx = (ring[:, 0].max() - ring[:, 0].min()) / 2.0
    rz = (ring[:, 2].max() - ring[:, 2].min()) / 2.0
    return (float(max(rx, 0.05)), float(max(rz, 0.03)))


def generate_body_smpl(
    measurements: BodyMeasurements,
) -> tuple[trimesh.Trimesh, BodyLandmarks, SkinningData]:
    """
    Generate SMPL body mesh, landmarks, and skinning data.

    Returns
    -------
    mesh : trimesh.Trimesh
        Body mesh (Y-up, feet at y=0).
    landmarks : BodyLandmarks
        Body landmark positions for garment fitting.
    skinning_data : SkinningData
        Weights, kintree parents, and joint positions for skeletal animation.
    """
    model_path = _smpl_path_for_gender(measurements.gender)
    if model_path is None:
        raise FileNotFoundError(
            f"SMPL model not found. Place SMPL_MALE.pkl and SMPL_FEMALE.pkl in {_SMPL_MODEL_DIR}"
        )

    # Load model data (must be numpy-only pkl — run convert_smpl.py first)
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
    except ModuleNotFoundError as e:
        raise FileNotFoundError(
            f"SMPL pkl file '{model_path.name}' contains chumpy objects. "
            f"Run 'python convert_smpl.py' from the server/ directory to convert. "
            f"Original error: {e}"
        ) from e

    def _to_np(arr, dtype):
        a = np.asarray(arr, dtype=dtype)
        if a.size == 0:
            raise ValueError(f"Empty array for key in {model_path}")
        return a

    try:
        v_template = _to_np(data["v_template"], np.float64)   # (6890, 3)
        shapedirs = _to_np(data["shapedirs"], np.float64)    # (6890, 3, 300)
        faces = _to_np(data["f"], np.int64)                  # (13776, 3)
        joints_template = _to_np(data["J"], np.float64)     # (24, 3)
        weights = _to_np(data["weights"], np.float64)        # (6890, 24)
        kintree_table = _to_np(data["kintree_table"], np.int64)  # (2, 24)
    except KeyError as e:
        raise FileNotFoundError(
            f"SMPL pkl missing expected key {e}. Re-run convert_smpl.py to produce numpy-only pkl."
        ) from e

    # Mean shape (zero betas = average body)
    betas = np.zeros(10, dtype=np.float64)
    vertices = _lbs_forward(v_template, shapedirs, betas)

    # ------------------------------------------------------------------
    # SMPL coordinate system: Y is UP, centre roughly at pelvis.
    # We do NOT swap axes. Just scale and translate so feet are at y = 0.
    # ------------------------------------------------------------------
    y_min = float(vertices[:, 1].min())
    y_max = float(vertices[:, 1].max())
    mesh_height = y_max - y_min
    if mesh_height <= 0:
        mesh_height = 1.0

    target_height_m = measurements.height_cm / 100.0
    scale = target_height_m / mesh_height

    vertices *= scale
    joints_scaled = joints_template * scale

    # Translate so feet are at y = 0
    y_min_scaled = float(vertices[:, 1].min())
    vertices[:, 1] -= y_min_scaled
    joints_scaled[:, 1] -= y_min_scaled

    # ------------------------------------------------------------------
    # Extract body landmarks for garment positioning
    # ------------------------------------------------------------------
    j = {name: joints_scaled[i] for i, name in enumerate(_JOINT_NAMES)}

    shoulder_y = float((j["left_shoulder"][1] + j["right_shoulder"][1]) / 2.0)
    shoulder_half_w = float(abs(j["left_shoulder"][0] - j["right_shoulder"][0]) / 2.0)
    pelvis_y = float(j["pelvis"][1])
    neck_y = float(j["neck"][1])
    ankle_y = float((j["left_ankle"][1] + j["right_ankle"][1]) / 2.0)

    # Waist ~ midway between pelvis and spine2
    waist_y = float((j["pelvis"][1] + j["spine2"][1]) / 2.0)

    # Inseam = crotch height; approximate as pelvis_y (crotch is slightly below pelvis)
    inseam_y = float(pelvis_y * 0.90)

    # Arm radius: half the thickness of the upper arm ring
    elbow_y = float((j["left_elbow"][1] + j["right_elbow"][1]) / 2.0)
    arm_y = (shoulder_y + elbow_y) / 2.0
    arm_mask = (
        (np.abs(vertices[:, 1] - arm_y) < 0.04) &
        (vertices[:, 0] > shoulder_half_w * 0.5)
    )
    if arm_mask.sum() >= 3:
        arm_verts = vertices[arm_mask]
        arm_r = float((arm_verts[:, 2].max() - arm_verts[:, 2].min()) / 2.0)
    else:
        arm_r = target_height_m * 0.04

    # Cross-section radii at chest and waist
    chest_y = float((shoulder_y + neck_y) / 2.0)
    chest_rx, chest_rz = _estimate_cross_section_radii(vertices, chest_y)
    waist_rx, waist_rz = _estimate_cross_section_radii(vertices, waist_y)
    hip_rx, _ = _estimate_cross_section_radii(vertices, pelvis_y)

    landmarks: BodyLandmarks = {
        "shoulder_y": shoulder_y,
        "shoulder_half_w": shoulder_half_w,
        "waist_y": waist_y,
        "pelvis_y": pelvis_y,
        "hip_rx": hip_rx,
        "neck_y": neck_y,
        "inseam_y": inseam_y,
        "ankle_y": ankle_y,
        "arm_r": arm_r,
        "chest_rx": chest_rx,
        "chest_rz": chest_rz,
        "waist_rx": waist_rx,
        "waist_rz": waist_rz,
    }

    # ------------------------------------------------------------------
    # Build mesh, colour, fix normals
    # ------------------------------------------------------------------
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    apply_face_color(mesh, _SKIN_COLOR)
    trimesh.repair.fix_normals(mesh)
    _ = mesh.vertex_normals  # force smooth normal computation

    # ------------------------------------------------------------------
    # Skinning data for animation
    # ------------------------------------------------------------------
    kintree_parents = kintree_table[0].copy()
    # Root joint (pelvis) parent is typically -1 in SMPL; ensure it's -1
    kintree_parents[0] = -1

    skinning: SkinningData = {
        "weights": weights,
        "kintree_parents": kintree_parents,
        "joints_positions": joints_scaled,
    }

    return mesh, landmarks, skinning
