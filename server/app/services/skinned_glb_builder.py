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
import logging
from typing import Sequence

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# SMPL joint names (must match smpl_body._JOINT_NAMES)
_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]

# Default skin color (222, 195, 170) normalized to [0-1]
_SKIN_COLOR_FACTOR = [222 / 255, 195 / 255, 170 / 255, 1.0]


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


def build_skinned_glb(
    mesh: trimesh.Trimesh,
    weights: np.ndarray,
    kintree_parents: np.ndarray,
    joints_positions: np.ndarray,
    *,
    base_color_factor: list[float] | None = None,
    material_info: dict | None = None,
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
    # 2: indices (uint32)
    idx_data = faces.flatten().astype(np.uint32).tobytes()
    idx_off, idx_len = write_data(idx_data)
    # 3: joints_0 (uvec4 uint16)
    j0_off, j0_len = write_data(top4_indices_arr.tobytes())
    # 4: weights_0 (vec4 float32)
    w0_off, w0_len = write_data(top4_weights_arr.tobytes())
    # 5: inverse bind matrices (mat4 float32)
    # glTF stores matrices in column-major order; numpy .tobytes() is row-major.
    # Transpose each 4x4 so row-major serialisation produces correct column-major layout.
    ibm_col_major = np.ascontiguousarray(inverse_bind_matrices.transpose(0, 2, 1))
    ibm_off, ibm_len = write_data(ibm_col_major.tobytes())

    # Optional: vertex colors (vec4 float32)
    vc_off = vc_len = 0
    if mat_type == "vertex_colors":
        vc_data = mat_info["colors"].astype(np.float32)
        vc_off, vc_len = write_data(vc_data.tobytes())

    # Optional: UV coordinates (vec2 float32)
    uv_off = uv_len = 0
    if mat_type == "texture":
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

    # BV 2: indices
    bv_idx = add_buffer_view(idx_off, idx_len, ELEMENT_ARRAY_BUFFER)
    n_indices = faces.size
    acc_idx = add_accessor(bv_idx, UNSIGNED_INT, n_indices, SCALAR)

    # BV 3: joints_0
    bv_j0 = add_buffer_view(j0_off, j0_len, ARRAY_BUFFER)
    acc_j0 = add_accessor(bv_j0, UNSIGNED_SHORT, n_verts, VEC4)

    # BV 4: weights_0
    bv_w0 = add_buffer_view(w0_off, w0_len, ARRAY_BUFFER)
    acc_w0 = add_accessor(bv_w0, FLOAT, n_verts, VEC4)

    # BV 5: inverse bind matrices
    bv_ibm = add_buffer_view(ibm_off, ibm_len)
    acc_ibm = add_accessor(bv_ibm, FLOAT, n_joints, MAT4)

    # Optional: vertex colors accessor
    acc_vc = None
    if mat_type == "vertex_colors" and vc_len > 0:
        bv_vc = add_buffer_view(vc_off, vc_len, ARRAY_BUFFER)
        acc_vc = add_accessor(bv_vc, FLOAT, n_verts, VEC4)

    # Optional: UV accessor
    acc_uv = None
    if mat_type == "texture" and uv_len > 0:
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

    gltf.bufferViews = buffer_views
    gltf.accessors = accessors

    # === Material ===
    gltf.images = []
    gltf.samplers = []
    gltf.textures = []

    if mat_type == "texture" and acc_uv is not None:
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
    else:
        # Default: use base_color_factor param, or skin color
        color = base_color_factor or _SKIN_COLOR_FACTOR
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

    # Extract material info before modifying the mesh
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
