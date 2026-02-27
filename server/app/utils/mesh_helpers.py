"""Shared 3D mesh construction helpers.

Provides factory functions for parametric geometry primitives used
by both the body generator and garment processor services.
"""

from __future__ import annotations

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def smooth_step(t: float | np.ndarray) -> float | np.ndarray:
    """Hermite smooth-step interpolation (ease-in / ease-out).

    Maps *t* in [0, 1] to a smooth S-curve so that transitions between
    body-segment radii don't produce harsh edges.
    """
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Tapered cylinder (frustum)
# ---------------------------------------------------------------------------

def create_tapered_cylinder(
    radius_top: float,
    radius_bottom: float,
    height: float,
    sections: int = 16,
) -> trimesh.Trimesh:
    """Create a tapered cylinder (frustum) centred at the origin.

    Parameters
    ----------
    radius_top : float
        Radius at the top  (y = +height / 2).
    radius_bottom : float
        Radius at the bottom (y = -height / 2).
    height : float
        Total height along the Y axis.
    sections : int
        Number of radial subdivisions around the circumference.

    Returns
    -------
    trimesh.Trimesh
        A closed frustum mesh.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)

    top_y = height / 2.0
    bottom_y = -height / 2.0

    # Ring vertices -----------------------------------------------------------
    verts: list[list[float]] = []
    for angle in angles:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        verts.append([radius_bottom * cos_a, bottom_y, radius_bottom * sin_a])
    for angle in angles:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        verts.append([radius_top * cos_a, top_y, radius_top * sin_a])

    # Centre vertices for caps
    verts.append([0.0, bottom_y, 0.0])  # bottom centre
    verts.append([0.0, top_y, 0.0])     # top centre

    bottom_centre = len(verts) - 2
    top_centre = len(verts) - 1

    # Faces -------------------------------------------------------------------
    faces: list[list[int]] = []

    # Side quads (two tris each)
    for i in range(sections):
        ni = (i + 1) % sections
        b0, b1 = i, ni
        t0, t1 = i + sections, ni + sections
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])

    # Bottom cap
    for i in range(sections):
        faces.append([bottom_centre, (i + 1) % sections, i])

    # Top cap
    for i in range(sections):
        faces.append([top_centre, i + sections, (i + 1) % sections + sections])

    return trimesh.Trimesh(
        vertices=np.array(verts, dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Lofted tube (multiple cross-section rings)
# ---------------------------------------------------------------------------

def create_lofted_tube(
    radii_x: np.ndarray,
    radii_z: np.ndarray,
    heights: np.ndarray,
    ring_points: int = 32,
    cap_top: bool = True,
    cap_bottom: bool = True,
) -> trimesh.Trimesh:
    """Create a lofted mesh by sweeping elliptical cross-sections.

    Parameters
    ----------
    radii_x : ndarray of shape (N,)
        Side-to-side radius at each cross-section.
    radii_z : ndarray of shape (N,)
        Front-to-back radius at each cross-section.
    heights : ndarray of shape (N,)
        Y-position of each cross-section (ascending).
    ring_points : int
        Number of vertices per ring.
    cap_top, cap_bottom : bool
        Whether to add end-cap faces.

    Returns
    -------
    trimesh.Trimesh
    """
    n_sections = len(heights)
    angles = np.linspace(0.0, 2.0 * np.pi, ring_points, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Build vertex buffer: (n_sections * ring_points) vertices
    verts = np.empty((n_sections * ring_points, 3), dtype=np.float64)
    for i in range(n_sections):
        offset = i * ring_points
        verts[offset : offset + ring_points, 0] = radii_x[i] * cos_a
        verts[offset : offset + ring_points, 1] = heights[i]
        verts[offset : offset + ring_points, 2] = radii_z[i] * sin_a

    # Build face buffer: side quads between adjacent rings
    faces: list[list[int]] = []
    for i in range(n_sections - 1):
        for j in range(ring_points):
            nj = (j + 1) % ring_points
            v00 = i * ring_points + j
            v01 = i * ring_points + nj
            v10 = (i + 1) * ring_points + j
            v11 = (i + 1) * ring_points + nj
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])

    # Caps
    if cap_bottom:
        ci = len(verts)
        verts = np.vstack([verts, [[0.0, heights[0], 0.0]]])
        for j in range(ring_points):
            faces.append([ci, (j + 1) % ring_points, j])

    if cap_top:
        ci = len(verts)
        verts = np.vstack([verts, [[0.0, heights[-1], 0.0]]])
        top_start = (n_sections - 1) * ring_points
        for j in range(ring_points):
            faces.append([ci, top_start + j, top_start + (j + 1) % ring_points])

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=np.array(faces, dtype=np.int64),
    )
    # Ensure consistent outward-facing normals
    trimesh.repair.fix_normals(mesh)
    return mesh


# ---------------------------------------------------------------------------
# Lofted tube WITH UV coordinates (for garment texturing)
# ---------------------------------------------------------------------------

def create_lofted_tube_with_uvs(
    radii_x: np.ndarray,
    radii_z: np.ndarray,
    heights: np.ndarray,
    ring_points: int = 32,
) -> trimesh.Trimesh:
    """Create a lofted tube mesh with UV coordinates for texture mapping.

    UVs are assigned as:
      u = angular position  (0 -> 1 around the circumference)
      v = height position   (0 = bottom, 1 = top)

    The mesh is NOT capped so that the texture wraps cleanly.

    Parameters
    ----------
    radii_x, radii_z : ndarray of shape (N,)
        Elliptical radii at each cross-section.
    heights : ndarray of shape (N,)
        Y-positions (ascending).
    ring_points : int
        Vertices per ring.

    Returns
    -------
    trimesh.Trimesh
        Mesh with ``visual.uv`` populated.
    """
    n_sections = len(heights)
    # +1 points per ring so first and last share position but have u=0 / u=1
    n_ring = ring_points + 1
    angles = np.linspace(0.0, 2.0 * np.pi, n_ring)

    h_min, h_max = float(heights[0]), float(heights[-1])
    h_range = h_max - h_min if h_max != h_min else 1.0

    verts = np.empty((n_sections * n_ring, 3), dtype=np.float64)
    uvs = np.empty((n_sections * n_ring, 2), dtype=np.float64)

    for i in range(n_sections):
        offset = i * n_ring
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        verts[offset : offset + n_ring, 0] = radii_x[i] * cos_a
        verts[offset : offset + n_ring, 1] = heights[i]
        verts[offset : offset + n_ring, 2] = radii_z[i] * sin_a

        v_coord = (heights[i] - h_min) / h_range
        uvs[offset : offset + n_ring, 0] = np.linspace(0.0, 1.0, n_ring)
        uvs[offset : offset + n_ring, 1] = v_coord

    # Side faces
    faces: list[list[int]] = []
    for i in range(n_sections - 1):
        for j in range(ring_points):  # NOT n_ring: skip the seam duplicate
            v00 = i * n_ring + j
            v01 = i * n_ring + j + 1
            v10 = (i + 1) * n_ring + j
            v11 = (i + 1) * n_ring + j + 1
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=np.array(faces, dtype=np.int64),
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    return mesh


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def apply_face_color(
    mesh: trimesh.Trimesh,
    color: tuple[int, int, int, int],
) -> trimesh.Trimesh:
    """Set a uniform RGBA face colour on every face of *mesh*."""
    color_array = np.tile(
        np.array(color, dtype=np.uint8),
        (len(mesh.faces), 1),
    )
    mesh.visual.face_colors = color_array
    return mesh


def assign_cylindrical_uvs(mesh: trimesh.Trimesh, axis: int = 1) -> np.ndarray:
    """Assign cylindrical UVs for texture wrapping (Y-up by default).

    u = angle around axis (0 to 1), v = along-axis position (0 to 1).
    For axis=1 (Y-up), the front-facing band (vertices facing +Z, camera view) is
    mapped to U in [0.25, 0.75] so the center of the texture (main garment view)
    appears on the front with less stretch; sides/back use the remaining U range.

    Uses arctan2(z, -x) so the U direction matches camera-view left-to-right:
    camera-left → low U, camera-right → high U.  Without the -x negate the
    image would appear mirrored on the front of the mesh.

    Returns UV array of shape (n_vertices, 2) for use with TextureVisuals.
    """
    verts = mesh.vertices
    n = len(verts)
    uvs = np.empty((n, 2), dtype=np.float64)

    if axis == 1:  # Y up: front (+Z) band -> U [0.25, 0.75]
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        angle = np.arctan2(z, -x)  # negate x so U goes left-to-right in camera view
        front_lo, front_hi = np.pi / 4.0, 3.0 * np.pi / 4.0
        u = np.empty(n, dtype=np.float64)
        front_mask = (angle >= front_lo) & (angle <= front_hi)
        u[front_mask] = 0.25 + 0.5 * (angle[front_mask] - front_lo) / (front_hi - front_lo)
        left_mask = angle < front_lo
        u[left_mask] = 0.25 * (angle[left_mask] + np.pi) / (front_lo + np.pi)
        right_mask = angle > front_hi
        u[right_mask] = 0.75 + 0.25 * (angle[right_mask] - front_hi) / (np.pi - front_hi)
        uvs[:, 0] = u
        y_min, y_max = y.min(), y.max()
        # glTF V=0 is image TOP, so invert: mesh top (collar) → V=0 → image top
        uvs[:, 1] = 1.0 - (y - y_min) / max(y_max - y_min, 1e-6)
    elif axis == 0:
        y, z = verts[:, 1], verts[:, 2]
        uvs[:, 0] = (np.arctan2(z, y) / (2.0 * np.pi)) + 0.5
        x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
        uvs[:, 1] = (verts[:, 0] - x_min) / max(x_max - x_min, 1e-6)
    else:  # axis == 2
        x, y = verts[:, 0], verts[:, 1]
        uvs[:, 0] = (np.arctan2(y, x) / (2.0 * np.pi)) + 0.5
        z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
        uvs[:, 1] = (verts[:, 2] - z_min) / max(z_max - z_min, 1e-6)

    uvs = np.clip(uvs, 0.0, 1.0)

    # Laplacian smooth UVs along mesh edges to reduce distortion at
    # arm-torso junctions where the cylindrical projection creates
    # sharp UV transitions between neighboring vertices.
    try:
        edges = mesh.edges_unique
        if len(edges) > 0:
            for _ in range(3):
                new_uvs = uvs.copy()
                # Build per-vertex neighbor sum and count
                neighbor_sum = np.zeros_like(uvs)
                neighbor_count = np.zeros(n, dtype=np.float64)
                for e0, e1 in edges:
                    neighbor_sum[e0] += uvs[e1]
                    neighbor_sum[e1] += uvs[e0]
                    neighbor_count[e0] += 1
                    neighbor_count[e1] += 1
                has_neighbors = neighbor_count > 0
                avg = neighbor_sum[has_neighbors] / neighbor_count[has_neighbors, np.newaxis]
                # Blend 20% toward neighbor average (gentle smoothing)
                new_uvs[has_neighbors] = 0.8 * uvs[has_neighbors] + 0.2 * avg
                uvs = np.clip(new_uvs, 0.0, 1.0)
    except Exception:
        pass  # non-critical; use unsmoothed UVs

    return uvs


def assign_pants_uvs_simple(mesh: trimesh.Trimesh) -> np.ndarray:
    """Simple XY projection UVs for pants (no vertex duplication).

    Used for solid-colour textures where front/back distinction is
    irrelevant.  Does NOT modify the mesh.

    Returns UV array of shape (n_vertices, 2).
    """
    verts = mesh.vertices
    x, y = verts[:, 0], verts[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    u = (x - x_min) / max(x_max - x_min, 1e-6)
    v = 1.0 - (y - y_min) / max(y_max - y_min, 1e-6)
    return np.clip(np.column_stack([u, v]), 0.0, 1.0)


def assign_pants_uvs(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """UV mapping for bifurcated garments with proper front/back UV seam.

    Creates a **dual-half texture atlas**:

    * Front-facing faces (centroid Z ≥ median) → left half  (U 0.0–0.5)
    * Back-facing  faces (centroid Z <  median) → right half (U 0.5–1.0)

    Within each half, U is a linear XY projection so the garment photo
    maps naturally onto both legs without cylindrical distortion.

    Vertices shared by front and back faces are **duplicated** to create
    a clean UV seam (the standard technique for UV discontinuities).
    This prevents the GPU from stretching the texture across the entire
    atlas for faces that straddle the boundary.

    **Modifies the mesh in-place** (adds duplicate vertices, updates
    face indices).

    Returns
    -------
    uvs : ndarray of shape (n_new_vertices, 2)
    seam_source_indices : ndarray of original vertex indices for each
        duplicated vertex.  Empty array if no duplication occurred.
        The caller must use this to extend any per-vertex mappings
        (e.g. ``vertex_map`` for skinning).
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_orig = len(verts)

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    u_base = (x - x_min) / max(x_max - x_min, 1e-6)

    y_min, y_max = float(y.min()), float(y.max())
    v_coord = 1.0 - (y - y_min) / max(y_max - y_min, 1e-6)

    z_med = float(np.median(z))

    # ── Classify every face as front or back ──────────────────────────
    face_cz = z[faces].mean(axis=1)
    face_is_front = face_cz >= z_med

    # ── Find seam vertices (shared by both front and back faces) ──────
    vert_in_front = np.zeros(n_orig, dtype=bool)
    vert_in_back = np.zeros(n_orig, dtype=bool)
    if face_is_front.any():
        vert_in_front[np.unique(faces[face_is_front])] = True
    if (~face_is_front).any():
        vert_in_back[np.unique(faces[~face_is_front])] = True

    seam_mask = vert_in_front & vert_in_back
    seam_indices = np.where(seam_mask)[0]
    n_seam = len(seam_indices)

    # ── Duplicate seam vertices ───────────────────────────────────────
    dup_start = n_orig
    remap = np.arange(n_orig, dtype=np.int64)
    if n_seam > 0:
        remap[seam_indices] = np.arange(dup_start, dup_start + n_seam, dtype=np.int64)

    new_faces = faces.copy()
    back_mask = ~face_is_front
    if back_mask.any():
        new_faces[back_mask] = remap[faces[back_mask]]

    if n_seam > 0:
        new_verts = np.vstack([verts, verts[seam_indices]])
    else:
        new_verts = verts.copy()

    # ── Build UV array ────────────────────────────────────────────────
    n_total = len(new_verts)
    uvs = np.zeros((n_total, 2))

    # Default: front UVs for all original vertices
    uvs[:n_orig, 0] = u_base * 0.5
    uvs[:n_orig, 1] = v_coord

    # Override back-only vertices to right half
    back_only = vert_in_back & ~vert_in_front
    back_only_idx = np.where(back_only)[0]
    if len(back_only_idx) > 0:
        uvs[back_only_idx, 0] = 0.5 + u_base[back_only_idx] * 0.5

    # Duplicated seam vertices → right half (back UVs)
    if n_seam > 0:
        uvs[dup_start:dup_start + n_seam, 0] = 0.5 + u_base[seam_indices] * 0.5
        uvs[dup_start:dup_start + n_seam, 1] = v_coord[seam_indices]

    # ── Apply to mesh ─────────────────────────────────────────────────
    mesh.vertices = new_verts
    mesh.faces = new_faces

    return np.clip(uvs, 0.0, 1.0), seam_indices


def assign_front_projection_uvs(mesh: trimesh.Trimesh) -> np.ndarray:
    """Assign UVs by projecting vertices onto the XY plane (front view).

    This is ideal for conforming garments where the 2D garment image is a
    front view — the projection maps the texture correctly onto the front
    of the garment.

    Returns UV array of shape (n_vertices, 2).
    """
    verts = mesh.vertices
    x, y = verts[:, 0], verts[:, 1]
    u = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
    u = 1.0 - u  # flip so left-right matches image orientation
    v = (y - y.min()) / max(float(y.max() - y.min()), 1e-6)
    return np.clip(np.column_stack([u, v]), 0.0, 1.0)
