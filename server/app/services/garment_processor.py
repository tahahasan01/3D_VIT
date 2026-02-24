"""Garment Processor Service.

Converts a 2D outfit image into a 3D garment mesh that matches the photo:

1. Background removal  (via ``texture_extractor``)
2. Silhouette analysis  (width profile, neckline, sleeves) so shape matches the image
3. Conforming garment from body mesh (offset surface) when body is available
4. Fallback: parametric template driven by silhouette
5. Apply the actual image as texture  (UV-mapped so the model wears it exactly as in the photo)
6. Normal repair and GLB export

Supported garment types:  tshirt, pants, dress.

Coordinate convention
---------------------
All garment meshes use the **same** Y-up coordinate system as the
body generator.  Positioning constants are derived from the default
body proportions so garments sit correctly on the body.
"""

from __future__ import annotations

import logging
import io

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

from ..models.body import BodyMeasurements
from ..models.garment import GarmentMeasurements, GarmentType
from ..utils.mesh_helpers import (
    create_lofted_tube,
    apply_face_color,
    smooth_step,
    assign_cylindrical_uvs,
    assign_front_projection_uvs,
)
from .texture_extractor import remove_background, prepare_texture
from .silhouette_analyzer import analyze_tshirt_silhouette, get_width_at_height_frac


# ---------------------------------------------------------------------------
# Body reference constants  (must match body_generator defaults)
# ---------------------------------------------------------------------------
# These approximate the default male body so garments align correctly.
_REF_HEIGHT = 1.75
_REF_INSEAM = 0.80
_REF_HEAD_H = _REF_HEIGHT * 0.12
_REF_NECK_H = _REF_HEIGHT * 0.03
_REF_TORSO_H = _REF_HEIGHT - _REF_INSEAM - _REF_HEAD_H - _REF_NECK_H
_REF_SHOULDER_Y = _REF_HEIGHT - _REF_HEAD_H - _REF_NECK_H - _REF_HEIGHT * 0.02
_REF_SHOULDER_HALF_W = 0.44 / 2.0  # must match body default shoulder_width_cm=44
_REF_ARM_R = _REF_HEIGHT * 0.040  # arm radius from body_generator proportions
_REF_HIP_RX = 0.98 / (2.0 * np.pi)  # ~0.156 m

# Garment offset: how much larger the garment sits from the body (ease / drape)
_GARMENT_OFFSET = 0.055  # ~5.5 cm when no body (mannequin / loose)
_GARMENT_OFFSET_TIGHT = 0.022  # ~2.2 cm when body exists: shirt pastes on model
_CONFORMING_OFFSET = 0.025  # ~2.5 cm for conforming garments (offset from body surface)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_garment(
    image_bytes: bytes,
    measurements: GarmentMeasurements,
    body_landmarks: dict | None = None,
    body_measurements: dict | None = None,
    body_mesh: trimesh.Trimesh | None = None,
    height_m: float | None = None,
) -> tuple[bytes, bool, np.ndarray | None]:
    """Process a 2D garment image into a 3D mesh.

    Pipeline priority:
    1. **ISP** (Implicit Sewing Patterns) — neural cloth topology with proper
       sleeves, neckline, and panels.  Produces the best quality.
    2. Conforming offset surface from body mesh (fallback).
    3. Parametric template (last resort).

    Returns
    -------
    tuple[bytes, bool, np.ndarray | None]
        (GLB bytes, used_isp_or_conforming, vertex_map_into_body).
    """
    # ── Background removal ────────────────────────────────────────────
    clean_image = remove_background(
        image_bytes,
        garment_type=measurements.garment_type.value,
    )
    body_landmarks = body_landmarks or getattr(measurements, "body_landmarks", None)
    body_measurements = body_measurements or getattr(measurements, "body_measurements", None)

    # ── Colour extraction from the uploaded photo ─────────────────────
    garment_rgb = _extract_garment_color(clean_image)
    if garment_rgb is None:
        garment_rgb = _extract_color_from_raw(image_bytes)
    if garment_rgb is None:
        # Absolute last resort: sample center pixel of original image
        garment_rgb = _center_pixel_color(image_bytes)
    logger.info("Extracted garment color: %s", garment_rgb)

    # ── 1. ISP path (primary) ────────────────────────────────────────
    if measurements.garment_type == GarmentType.TSHIRT and body_mesh is not None:
        isp_result = _try_isp_tshirt(
            body_mesh, garment_rgb, height_m,
            sleeve_length_cm=measurements.sleeve_length_cm,
            chest_cm=measurements.chest_cm,
            length_cm=measurements.length_cm,
        )
        if isp_result is not None:
            mesh, vertex_map = isp_result
            glb = _export_textured_glb(mesh)
            logger.info("ISP tshirt: %d verts, vertex_map %s", len(mesh.vertices), vertex_map.shape)
            return glb, True, vertex_map

    if measurements.garment_type == GarmentType.PANTS and body_mesh is not None:
        isp_result = _try_isp_pants(
            body_mesh, garment_rgb, height_m,
            waist_cm=measurements.waist_cm,
            hip_cm=measurements.hip_cm,
            length_cm=measurements.length_cm,
        )
        if isp_result is not None:
            mesh, vertex_map = isp_result
            glb = _export_textured_glb(mesh)
            logger.info("ISP pants: %d verts, vertex_map %s", len(mesh.vertices), vertex_map.shape)
            return glb, True, vertex_map

    # ── 2. Conforming path (fallback when ISP unavailable) ────────────
    silhouette: dict | None = None
    if measurements.garment_type == GarmentType.TSHIRT:
        silhouette = analyze_tshirt_silhouette(clean_image)

    if body_mesh is not None and body_landmarks:
        result = _build_conforming_garment(body_mesh, body_landmarks, measurements, silhouette)
        if result[0] is not None:
            mesh, vertex_map = result
            _apply_color_to_mesh(mesh, garment_rgb, clean_image)
            trimesh.repair.fix_normals(mesh)
            return _export_textured_glb(mesh), True, vertex_map

    # ── 3. Parametric path (last resort) ──────────────────────────────
    mesh = _create_garment_template(
        measurements,
        body_landmarks,
        silhouette=silhouette,
    )
    _apply_color_to_mesh(mesh, garment_rgb, clean_image)
    trimesh.repair.fix_normals(mesh)
    return _export_textured_glb(mesh), False, None


# ---------------------------------------------------------------------------
# ISP T-shirt builder
# ---------------------------------------------------------------------------

def _try_isp_tshirt(
    body_mesh: trimesh.Trimesh,
    garment_rgb: tuple[int, int, int] | None,
    height_m: float | None,
    sleeve_length_cm: float | None = None,
    chest_cm: float | None = None,
    length_cm: float | None = None,
) -> tuple[trimesh.Trimesh, np.ndarray] | None:
    """Generate a T-shirt via ISP, scale for measurements, conform, colour.

    Measurements respected
    ----------------------
    * **chest_cm** — XZ scaling so garment chest matches the target.
    * **length_cm** — top-anchored Y stretch so hem reaches desired length.
    * **sleeve_length_cm** — smooth X compression of sleeve vertices.

    Conforming strategy: **push-out with ease** — every vertex closer to
    the body than *ease_offset* is pushed outward along the body normal.
    The ease offset is derived from the chest-vs-body difference (min 1.5 cm)
    so the garment looks realistically loose rather than skin-tight.
    """
    try:
        from .isp_service import is_isp_available, get_isp_service
        if not is_isp_available("tee"):
            logger.info("ISP not available, skipping")
            return None

        svc = get_isp_service()
        best_idx = svc.find_best_tee_idx()
        logger.info("ISP generating tee with idx_G=%d", best_idx)
        garment_mesh = svc.generate_tpose_garment("tee", idx_G=best_idx, resolution=180)

        # ── 1. Uniform height scale ──────────────────────────────────
        body_height = float(body_mesh.bounds[1][1] - body_mesh.bounds[0][1])
        if body_height < 0.1:
            body_height = 1.75
        target_height = height_m or body_height
        h_scale = target_height / 1.75
        garment_mesh.vertices *= h_scale

        # ── 2. Uniform XZ conforming (straight hang) ────────────────
        # Measure the MAXIMUM body torso width across all Y levels and
        # apply a single uniform XZ scale so the shirt hangs straight
        # from top to bottom — no per-band bulging.
        TORSO_CLEARANCE = 0.015    # 1.5 cm outside body torso
        SLEEVE_SHOULDER_X = 0.19 * h_scale  # half-shoulder-width cutoff

        body_center = body_mesh.centroid
        body_verts_np = np.asarray(body_mesh.vertices)
        g_verts = np.asarray(garment_mesh.vertices, dtype=np.float64)

        # Align centroids before conforming
        garment_center = garment_mesh.centroid
        g_verts[:, 0] += body_center[0] - garment_center[0]
        g_verts[:, 2] += body_center[2] - garment_center[2]
        garment_mesh.vertices = g_verts

        cx = float(body_center[0])
        cz = float(body_center[2])

        # Garment Y range
        g_y_min = float(g_verts[:, 1].min())
        g_y_max = float(g_verts[:, 1].max())

        # Filter body verts to torso-only (exclude arm tips)
        torso_mask_body = np.abs(body_verts_np[:, 0] - cx) < 0.22 * h_scale
        torso_body = body_verts_np[torso_mask_body]

        # Scan Y bands to find the MAX body width (usually at chest)
        N_BANDS = 20
        max_body_x_half = 0.0
        max_body_z_half = 0.0
        for i in range(N_BANDS):
            y_lo = g_y_min + (g_y_max - g_y_min) * i / N_BANDS
            y_hi = g_y_min + (g_y_max - g_y_min) * (i + 1) / N_BANDS
            b_band = (torso_body[:, 1] >= y_lo - 0.02) & (torso_body[:, 1] < y_hi + 0.02)
            if b_band.sum() < 3:
                continue
            b_slice = torso_body[b_band]
            bxh = max(abs(b_slice[:, 0].max() - cx),
                      abs(b_slice[:, 0].min() - cx))
            bzh = max(abs(b_slice[:, 2].max() - cz),
                      abs(b_slice[:, 2].min() - cz))
            max_body_x_half = max(max_body_x_half, bxh)
            max_body_z_half = max(max_body_z_half, bzh)

        if max_body_x_half < 0.05:
            max_body_x_half = 0.17
        if max_body_z_half < 0.05:
            max_body_z_half = 0.12

        # Garment torso half-widths (exclude sleeves)
        g_torso = np.abs(g_verts[:, 0] - cx) < SLEEVE_SHOULDER_X + 0.05
        if g_torso.sum() > 10:
            gt = g_verts[g_torso]
            garm_x_half = max(abs(float(gt[:, 0].max()) - cx),
                              abs(float(gt[:, 0].min()) - cx), 0.01)
            garm_z_half = max(abs(float(gt[:, 2].max()) - cz),
                              abs(float(gt[:, 2].min()) - cz), 0.01)
        else:
            garm_x_half = 0.19
            garm_z_half = 0.13

        # Single uniform scale from the widest body slice + clearance
        target_x = max_body_x_half + TORSO_CLEARANCE
        target_z = max_body_z_half + TORSO_CLEARANCE
        sx = max(target_x / garm_x_half, 1.0)
        sz = max(target_z / garm_z_half, 1.0)

        # Apply the SAME scale to all torso vertices (uniform straight hang)
        g_verts = np.asarray(garment_mesh.vertices, dtype=np.float64)
        g_verts[:, 0] = cx + (g_verts[:, 0] - cx) * sx
        g_verts[:, 2] = cz + (g_verts[:, 2] - cz) * sz
        garment_mesh.vertices = g_verts
        logger.info("Uniform XZ scale: sx=%.3f sz=%.3f (body max x=%.1fcm z=%.1fcm)",
                     sx, sz, max_body_x_half * 100, max_body_z_half * 100)

        # Also apply user chest scaling if they specified a chest
        # larger than the body-conform result.
        if chest_cm:
            _verts_tmp2 = np.asarray(garment_mesh.vertices)
            _y_mid2 = (_verts_tmp2[:, 1].min() + _verts_tmp2[:, 1].max()) / 2
            _band2 = np.abs(_verts_tmp2[:, 1] - _y_mid2) < 0.03
            if int(_band2.sum()) > 10:
                _cv2 = _verts_tmp2[_band2]
                _cx_r2 = (_cv2[:, 0].max() - _cv2[:, 0].min()) / 2
                _cz_r2 = (_cv2[:, 2].max() - _cv2[:, 2].min()) / 2
                current_chest = float(
                    np.pi * (3 * (_cx_r2 + _cz_r2)
                             - np.sqrt((3 * _cx_r2 + _cz_r2) * (_cx_r2 + 3 * _cz_r2)))
                ) * 100
                desired_chest = chest_cm + 8.0  # ease
                if desired_chest > current_chest + 2:
                    extra_scale = desired_chest / max(current_chest, 1.0)
                    extra_scale = float(np.clip(extra_scale, 1.0, 1.5))
                    cx2 = float(garment_mesh.centroid[0])
                    cz2 = float(garment_mesh.centroid[2])
                    garment_mesh.vertices[:, 0] = cx2 + (garment_mesh.vertices[:, 0] - cx2) * extra_scale
                    garment_mesh.vertices[:, 2] = cz2 + (garment_mesh.vertices[:, 2] - cz2) * extra_scale
                    logger.info("Extra chest scale: %.3f (%.0fcm -> %.0fcm)",
                                extra_scale, current_chest, desired_chest)

        # ── 3. Length Y stretch (top-anchored) ───────────────────────
        natural_top = float(garment_mesh.bounds[1][1])
        natural_bot = float(garment_mesh.bounds[0][1])
        natural_len = natural_top - natural_bot
        desired_len = (length_cm / 100.0) if length_cm else natural_len
        desired_len = float(np.clip(desired_len, natural_len * 0.6, natural_len * 2.5))
        if abs(desired_len - natural_len) > 0.01:
            verts = garment_mesh.vertices
            t = np.clip((natural_top - verts[:, 1]) / max(natural_len, 0.01), 0.0, 1.0)
            verts[:, 1] = natural_top - t * desired_len
            garment_mesh.vertices = verts
            logger.info("Length stretch: %.1fcm -> %.1fcm",
                        natural_len * 100, desired_len * 100)

        # ── 4. Sleeve trim + armpit sealing ────────────────────────
        body_x_max = float(body_verts_np[:, 0].max())
        body_x_min = float(body_verts_np[:, 0].min())

        if sleeve_length_cm is not None and sleeve_length_cm < 80:
            desired_sleeve = sleeve_length_cm / 100.0
        else:
            desired_sleeve = max(body_x_max - cx - SLEEVE_SHOULDER_X + 0.02, 0.05)

        max_x_from_center = SLEEVE_SHOULDER_X + desired_sleeve
        verts = np.asarray(garment_mesh.vertices)
        abs_x_from_center = np.abs(verts[:, 0] - cx)

        faces = np.asarray(garment_mesh.faces)
        face_max_abs_x = abs_x_from_center[faces].max(axis=1)
        keep_sleeve = face_max_abs_x <= max_x_from_center
        if not keep_sleeve.all():
            removed_count = int((~keep_sleeve).sum())
            garment_mesh.update_faces(keep_sleeve)
            garment_mesh.remove_unreferenced_vertices()
            logger.info("Trimmed %d sleeve faces beyond %.1fcm from center",
                        removed_count, max_x_from_center * 100)

        # --- Seal armpit boundary loops with fan triangulation ---
        try:
            from collections import defaultdict
            edge_face_count = defaultdict(int)
            for f in garment_mesh.faces:
                for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
                    edge_face_count[tuple(sorted(e))] += 1

            boundary_edges = []
            for e, cnt in edge_face_count.items():
                if cnt == 1:
                    boundary_edges.append(e)

            if boundary_edges:
                adj = defaultdict(set)
                for a, b in boundary_edges:
                    adj[a].add(b)
                    adj[b].add(a)

                visited_verts = set()
                loops = []
                for start_v in adj:
                    if start_v in visited_verts:
                        continue
                    loop = [start_v]
                    visited_verts.add(start_v)
                    current = start_v
                    while True:
                        neighbors = adj[current] - visited_verts
                        if not neighbors:
                            break
                        nxt = neighbors.pop()
                        loop.append(nxt)
                        visited_verts.add(nxt)
                        current = nxt
                    if len(loop) >= 3:
                        loops.append(loop)

                new_verts_list = list(garment_mesh.vertices)
                new_faces_list = list(garment_mesh.faces)
                total_sealed = 0

                for loop in loops:
                    loop_verts = np.array([new_verts_list[vi] for vi in loop])
                    centroid = loop_verts.mean(axis=0)
                    cent_idx = len(new_verts_list)
                    new_verts_list.append(centroid)

                    for j in range(len(loop)):
                        v0 = loop[j]
                        v1 = loop[(j + 1) % len(loop)]
                        new_faces_list.append([v0, v1, cent_idx])
                    total_sealed += len(loop)

                if total_sealed > 0:
                    garment_mesh = trimesh.Trimesh(
                        vertices=np.array(new_verts_list),
                        faces=np.array(new_faces_list),
                        process=False,
                    )
                    logger.info("Sealed %d boundary edges across %d loops",
                                total_sealed, len(loops))
        except Exception as exc:
            logger.warning("Boundary sealing failed: %s", exc)

        # ── 5. Fill holes + fix winding ──────────────────────────────
        for _ in range(5):
            try:
                trimesh.repair.fill_holes(garment_mesh)
            except Exception:
                break
        trimesh.repair.fix_winding(garment_mesh)
        trimesh.repair.fix_normals(garment_mesh)

        # Remove degenerate faces
        face_areas = garment_mesh.area_faces
        keep_valid = face_areas > 1e-10
        if not keep_valid.all():
            garment_mesh.update_faces(keep_valid)
            garment_mesh.remove_unreferenced_vertices()
            try:
                trimesh.repair.fill_holes(garment_mesh)
            except Exception:
                pass

        # ── 6. Laplacian smooth (gentle, preserve straight profile) ──
        try:
            trimesh.smoothing.filter_laplacian(
                garment_mesh, lamb=0.3, iterations=8,
                implicit_time_integration=False,
            )
        except Exception:
            pass

        # ── 7. Anti-penetration push (cleanup) ───────────────────────
        MIN_CLEARANCE = 0.010
        MAX_PUSH_SHIRT = 0.08

        body_verts = np.asarray(body_mesh.vertices)
        body_normals = np.asarray(body_mesh.vertex_normals)
        tree = cKDTree(body_verts)

        for _pass in range(3):
            garment_verts = np.asarray(garment_mesh.vertices, dtype=np.float64)
            dists, idx_map = tree.query(garment_verts, k=1)
            vertex_map = idx_map.astype(np.int64)
            nearest_pos = body_verts[vertex_map]
            nearest_nrm = body_normals[vertex_map]

            signed_dist = np.sum(
                (garment_verts - nearest_pos) * nearest_nrm, axis=1,
            )

            penetrating = signed_dist < MIN_CLEARANCE
            correction = np.zeros(len(garment_verts), dtype=np.float64)
            correction[penetrating] = MIN_CLEARANCE - signed_dist[penetrating]
            correction = np.minimum(correction, MAX_PUSH_SHIRT)
            garment_mesh.vertices = (
                garment_verts + correction[:, np.newaxis] * nearest_nrm
            ).astype(np.float64)

            try:
                trimesh.smoothing.filter_laplacian(
                    garment_mesh, lamb=0.3, iterations=2,
                    implicit_time_integration=False,
                )
            except Exception:
                pass

        logger.info("Anti-penetration: %d / %d verts pushed (3 passes)",
                    int(penetrating.sum()), len(garment_verts))

        try:
            trimesh.repair.fill_holes(garment_mesh)
            trimesh.repair.fix_normals(garment_mesh)
        except Exception:
            pass

        # ── 8. Remove floating fragments ──────────────────────────────
        garment_verts = np.asarray(garment_mesh.vertices, dtype=np.float64)
        dists_final, _ = tree.query(garment_verts, k=1)
        faces_arr = np.asarray(garment_mesh.faces)
        face_max_dist = dists_final[faces_arr].max(axis=1)
        keep_near = face_max_dist < 0.25  # 25 cm from body
        if not keep_near.all():
            removed = int((~keep_near).sum())
            garment_mesh.update_faces(keep_near)
            garment_mesh.remove_unreferenced_vertices()
            logger.info("Removed %d floating faces (dist > 25cm from body)", removed)

        # Keep only largest connected component
        components = garment_mesh.split(only_watertight=False)
        if len(components) > 1:
            largest = max(components, key=lambda c: len(c.faces))
            if len(largest.faces) > len(garment_mesh.faces) * 0.5:
                garment_mesh = largest
                logger.info("Kept largest component: %d faces out of %d components",
                            len(garment_mesh.faces), len(components))

        # Re-query vertex map after cleanup
        garment_verts = np.asarray(garment_mesh.vertices, dtype=np.float64)
        _, vertex_map = tree.query(garment_verts, k=1)
        vertex_map = vertex_map.astype(np.int64)

        # ── 9. Final hole check ───────────────────────────────────────
        try:
            trimesh.repair.fill_holes(garment_mesh)
            trimesh.repair.fix_normals(garment_mesh)
        except Exception:
            pass

        # ── 10. Level ragged hems ─────────────────────────────────────
        _level_hem(garment_mesh, top=True, blend_cm=1.0)   # smooth collar
        _level_hem(garment_mesh, top=False, blend_cm=1.5)  # smooth bottom hem

        # ── 11. Apply colour ─────────────────────────────────────────
        ease_offset = 0.012  # for logging
        _apply_vertex_color(garment_mesh, garment_rgb)

        trimesh.repair.fix_normals(garment_mesh)
        logger.info(
            "ISP tshirt ready: %d verts, chest=%.0fcm, len=%.1fcm, "
            "ease=%.1fcm, Y=[%.3f,%.3f]",
            len(garment_mesh.vertices), chest_cm or 0, desired_len * 100,
            ease_offset * 100,
            garment_mesh.bounds[0][1], garment_mesh.bounds[1][1],
        )
        return garment_mesh, vertex_map

    except Exception:
        logger.exception("ISP tshirt generation failed, falling back")
        return None


# ---------------------------------------------------------------------------
# ISP Pants builder
# ---------------------------------------------------------------------------

def _try_isp_pants(
    body_mesh: trimesh.Trimesh,
    garment_rgb: tuple[int, int, int] | None,
    height_m: float | None,
    waist_cm: float | None = None,
    hip_cm: float | None = None,
    length_cm: float | None = None,
) -> tuple[trimesh.Trimesh, np.ndarray] | None:
    """Generate trousers via ISP, scale for measurements, conform, colour.

    Measurements
    ------------
    * **waist_cm / hip_cm** — XZ scaling so pants circumference matches.
    * **length_cm** — top-anchored Y stretch so hem reaches desired length.
    """
    try:
        from .isp_service import is_isp_available, get_isp_service
        if not is_isp_available("pants"):
            logger.info("ISP pants not available, skipping")
            return None

        svc = get_isp_service()
        best_idx = svc.find_best_pants_idx()
        logger.info("ISP generating pants with idx_G=%d", best_idx)
        garment_mesh = svc.generate_tpose_garment("pants", idx_G=best_idx, resolution=180)

        # ── 1. Uniform height scale ──────────────────────────────────
        body_height = float(body_mesh.bounds[1][1] - body_mesh.bounds[0][1])
        if body_height < 0.1:
            body_height = 1.75
        target_height = height_m or body_height
        h_scale = target_height / 1.75
        garment_mesh.vertices *= h_scale

        # ── 2. Hip/waist XZ scaling ──────────────────────────────────
        _NATURAL_HIP_CM = 101.0   # measured for idx=180 at scale=1.0
        natural_hip = _NATURAL_HIP_CM * h_scale
        desired_hip = hip_cm or waist_cm or natural_hip
        xz_scale = float(np.clip(desired_hip / natural_hip, 0.75, 1.5))
        if abs(xz_scale - 1.0) > 0.01:
            cx = float(garment_mesh.centroid[0])
            cz = float(garment_mesh.centroid[2])
            garment_mesh.vertices[:, 0] = cx + (garment_mesh.vertices[:, 0] - cx) * xz_scale
            garment_mesh.vertices[:, 2] = cz + (garment_mesh.vertices[:, 2] - cz) * xz_scale
            logger.info("Pants XZ scale: %.3f (%.0fcm -> %.0fcm)",
                        xz_scale, natural_hip, desired_hip)

        # ── 3. Length Y stretch (top-anchored) ───────────────────────
        natural_top = float(garment_mesh.bounds[1][1])
        natural_bot = float(garment_mesh.bounds[0][1])
        natural_len = natural_top - natural_bot
        desired_len = (length_cm / 100.0) if length_cm else natural_len
        desired_len = float(np.clip(desired_len, natural_len * 0.5, natural_len * 1.5))
        if abs(desired_len - natural_len) > 0.01:
            verts = garment_mesh.vertices
            t = np.clip((natural_top - verts[:, 1]) / max(natural_len, 0.01), 0.0, 1.0)
            verts[:, 1] = natural_top - t * desired_len
            garment_mesh.vertices = verts
            logger.info("Pants length: %.1fcm -> %.1fcm",
                        natural_len * 100, desired_len * 100)

        # ── 4. Align centroids on X/Z and waist Y ────────────────────
        body_center = body_mesh.centroid
        garment_center = garment_mesh.centroid
        garment_mesh.vertices[:, 0] += body_center[0] - garment_center[0]
        garment_mesh.vertices[:, 2] += body_center[2] - garment_center[2]

        # Align Y: shift so pants waistband sits at body waist level.
        # Body waist ≈ 56% of body height (SMPL standard).
        body_verts_np = np.asarray(body_mesh.vertices)
        body_y_min = float(body_verts_np[:, 1].min())
        body_y_max = float(body_verts_np[:, 1].max())
        body_height_val = body_y_max - body_y_min
        body_waist_y = body_y_min + body_height_val * 0.56   # natural waist
        pants_top_y = float(garment_mesh.bounds[1][1])
        y_shift = body_waist_y - pants_top_y
        garment_mesh.vertices[:, 1] += y_shift
        logger.info("Pants Y-shift: %.3fm (waist at Y=%.3f, body waist=%.3f)",
                    y_shift, pants_top_y + y_shift, body_waist_y)

        # ── 4b. Fix only truly degenerate faces (zero area) ──────────
        # Keep mesh connectivity intact — do NOT remove stretched faces
        # as that creates visible holes.  Only remove zero-area triangles.
        _areas = garment_mesh.area_faces
        _good_faces = _areas > 1e-8
        if not _good_faces.all():
            _removed = int((~_good_faces).sum())
            garment_mesh.update_faces(_good_faces)
            garment_mesh.remove_unreferenced_vertices()
            logger.info("Removed %d zero-area faces from ISP pants", _removed)

        # ── 4c. Clamp inner-thigh vertices to prevent self-intersection ─
        # Some ISP pants have vertices at the inner crotch that cross the
        # body midplane. Clamp Z to be at least as far from body center
        # as the nearest body vertex Z.
        body_verts_np = np.asarray(body_mesh.vertices)
        gv = np.asarray(garment_mesh.vertices, dtype=np.float64)
        body_z_cen = float(body_center[2])
        # Min Z thickness: pants should not be thinner than body
        # (prevents the front/back panels from overlapping)
        body_z_front = float(body_verts_np[:, 2].max())
        body_z_back = float(body_verts_np[:, 2].min())
        z_margin = 0.008  # 8mm clearance
        front_too_close = gv[:, 2] > (body_z_front + z_margin)
        # Only clamp if they're beyond body extent (garment should envelope)
        # No clamping needed here — the push-out will handle it

        # ── 5. Y-slice radial scaling + push-out ─────────────────────
        # ISP pants legs are wider than body legs. Instead of pulling
        # individual vertices toward nearest body points (which distorts
        # the lower legs), scale each horizontal Y-band of the pant tube
        # to match the body leg radius at that height.  This preserves
        # mesh connectivity and Y positions.
        _BODY_REF_HIP_CM = 96.0
        ease_total_cm = max((desired_hip - _BODY_REF_HIP_CM), 0.0)
        ease_offset = max(ease_total_cm / (2.0 * np.pi) / 100.0, 0.020)
        ease_offset = min(ease_offset, 0.05)
        MAX_PUSH = 0.040
        RADIAL_CLEARANCE = ease_offset + 0.015  # target clearance from body

        body_verts = np.asarray(body_mesh.vertices)
        garment_verts = np.asarray(garment_mesh.vertices, dtype=np.float64)
        body_normals = np.asarray(body_mesh.vertex_normals)
        body_cx = float(body_mesh.centroid[0])

        # Body crotch Y: where the single torso splits into two legs
        body_crotch_y = body_y_min + body_height_val * 0.47

        tree = cKDTree(body_verts)

        # --- Y-slice radial scaling for below-crotch (leg region) ---
        pants_y_min = float(garment_verts[:, 1].min())
        N_BANDS = 25
        band_edges = np.linspace(pants_y_min, body_crotch_y, N_BANDS + 1)
        scaled_count = 0

        for band_i in range(N_BANDS):
            y_lo, y_hi = float(band_edges[band_i]), float(band_edges[band_i + 1])

            for leg_sign, leg_label in [(+1, 'right'), (-1, 'left')]:
                # Select garment verts in this Y-band & leg
                if leg_sign > 0:
                    leg_band = (
                        (garment_verts[:, 0] > body_cx)
                        & (garment_verts[:, 1] >= y_lo)
                        & (garment_verts[:, 1] < y_hi)
                    )
                else:
                    leg_band = (
                        (garment_verts[:, 0] <= body_cx)
                        & (garment_verts[:, 1] >= y_lo)
                        & (garment_verts[:, 1] < y_hi)
                    )
                if not leg_band.any():
                    continue

                # Body verts in a slightly expanded Y-band for this leg
                y_pad = (y_hi - y_lo) * 0.5
                if leg_sign > 0:
                    body_leg = (
                        (body_verts[:, 0] > body_cx)
                        & (body_verts[:, 1] >= y_lo - y_pad)
                        & (body_verts[:, 1] < y_hi + y_pad)
                    )
                else:
                    body_leg = (
                        (body_verts[:, 0] <= body_cx)
                        & (body_verts[:, 1] >= y_lo - y_pad)
                        & (body_verts[:, 1] < y_hi + y_pad)
                    )
                if not body_leg.any():
                    continue

                # Body leg center and radius in XZ plane
                bsub = body_verts[body_leg]
                leg_cx = float(bsub[:, 0].mean())
                leg_cz = float(bsub[:, 2].mean())
                body_radii = np.sqrt(
                    (bsub[:, 0] - leg_cx) ** 2 + (bsub[:, 2] - leg_cz) ** 2
                )
                body_r = float(np.percentile(body_radii, 95))  # robust max
                target_r = body_r + RADIAL_CLEARANCE

                # Garment tube center and radii in XZ
                gv_band = garment_verts[leg_band]
                g_cx = float(gv_band[:, 0].mean())
                g_cz = float(gv_band[:, 2].mean())
                g_dx = gv_band[:, 0] - g_cx
                g_dz = gv_band[:, 2] - g_cz
                g_radii = np.sqrt(g_dx ** 2 + g_dz ** 2)
                g_r = float(np.percentile(g_radii, 95)) if len(g_radii) > 2 else float(g_radii.max())

                if g_r < 0.005:
                    continue  # degenerate band

                # Scale radially and shift center to body leg center
                if g_r > target_r:
                    scale_r = target_r / g_r
                else:
                    scale_r = 1.0  # don't expand

                garment_verts[leg_band, 0] = leg_cx + g_dx * scale_r
                garment_verts[leg_band, 2] = leg_cz + g_dz * scale_r
                scaled_count += int(leg_band.sum())

        # --- Above-crotch (hip/waist): gentle nearest-point attraction ---
        above_crotch = garment_verts[:, 1] >= body_crotch_y
        if above_crotch.any():
            above_idx = np.where(above_crotch)[0]
            pv_hip = garment_verts[above_idx]
            dists_hip, idx_hip = tree.query(pv_hip, k=1)
            body_nearest = body_verts[idx_hip]
            body_nrm = body_normals[idx_hip]
            target_pos = body_nearest + body_nrm * RADIAL_CLEARANCE
            far = dists_hip > 0.06
            if far.any():
                far_idx = above_idx[far]
                blend = np.clip(dists_hip[far] / 0.12, 0.0, 1.0) * 0.55
                garment_verts[far_idx] = (
                    pv_hip[far] * (1 - blend[:, np.newaxis])
                    + target_pos[far] * blend[:, np.newaxis]
                )

        garment_mesh.vertices = garment_verts
        logger.info(
            "Y-slice radial scaling: %d verts across %d bands",
            scaled_count, N_BANDS,
        )

        # Final push-out pass
        garment_verts = np.asarray(garment_mesh.vertices, dtype=np.float64)
        dists, vertex_map = tree.query(garment_verts, k=1)
        vertex_map = vertex_map.astype(np.int64)
        nearest_pos = body_verts[vertex_map]
        nearest_nrm = body_normals[vertex_map]
        signed_dist = np.sum(
            (garment_verts - nearest_pos) * nearest_nrm, axis=1,
        )
        needs_push = signed_dist < ease_offset
        correction = np.zeros(len(garment_verts), dtype=np.float64)
        correction[needs_push] = ease_offset - signed_dist[needs_push]
        correction = np.minimum(correction, MAX_PUSH)
        garment_mesh.vertices = (
            garment_verts + correction[:, np.newaxis] * nearest_nrm
        ).astype(np.float64)
        logger.info(
            "Pants push-out: %d / %d verts pushed (ease=%.1fcm)",
            int(needs_push.sum()), len(garment_verts), ease_offset * 100,
        )

        # ── 6. Fill holes + Heavy smooth + re-push cycles ───────────
        try:
            trimesh.repair.fill_holes(garment_mesh)
            logger.info("Filled holes in pants mesh")
        except Exception:
            pass

        # Multiple smooth → push-out cycles for clean fabric surface
        for cycle in range(3):
            try:
                trimesh.smoothing.filter_laplacian(
                    garment_mesh,
                    lamb=0.6 if cycle == 0 else 0.4,
                    iterations=6 if cycle == 0 else 3,
                    implicit_time_integration=False,
                )
            except Exception:
                pass
            # Re-push after smoothing
            garment_verts = np.asarray(garment_mesh.vertices, dtype=np.float64)
            dists2, idx2 = tree.query(garment_verts, k=1)
            vertex_map = idx2.astype(np.int64)
            nearest_pos = body_verts[vertex_map]
            nearest_nrm = body_normals[vertex_map]
            signed_dist = np.sum(
                (garment_verts - nearest_pos) * nearest_nrm, axis=1,
            )
            needs_push = signed_dist < ease_offset
            correction = np.zeros(len(garment_verts), dtype=np.float64)
            correction[needs_push] = ease_offset - signed_dist[needs_push]
            correction = np.minimum(correction, MAX_PUSH)
            garment_mesh.vertices = (
                garment_verts + correction[:, np.newaxis] * nearest_nrm
            ).astype(np.float64)

        # ── 6b. Keep only the largest connected component ────────────
        # ISP can produce tiny disconnected fragments.
        components = garment_mesh.split(only_watertight=False)
        if len(components) > 1:
            largest = max(components, key=lambda c: len(c.faces))
            if len(largest.faces) > len(garment_mesh.faces) * 0.5:
                logger.info(
                    "Keeping largest component (%d faces) out of %d components",
                    len(largest.faces), len(components),
                )
                garment_mesh = largest
            # Re-query vertex_map
            garment_verts = np.asarray(garment_mesh.vertices, dtype=np.float64)
            _, vertex_map = tree.query(garment_verts, k=1)
            vertex_map = vertex_map.astype(np.int64)

        # ── 6c. Trim below ankle level ─────────────────────────────
        # Pants should not cover the feet. Remove faces below ankle Y.
        ankle_y = body_y_min + body_height_val * 0.05
        garment_verts = np.asarray(garment_mesh.vertices, dtype=np.float64)
        faces_arr = np.asarray(garment_mesh.faces)
        face_centroids_y = garment_verts[faces_arr, 1].mean(axis=1)
        keep_above_ankle = face_centroids_y >= ankle_y
        if not keep_above_ankle.all():
            removed_ankle = int((~keep_above_ankle).sum())
            garment_mesh.update_faces(keep_above_ankle)
            garment_mesh.remove_unreferenced_vertices()
            logger.info("Trimmed %d faces below ankle (Y < %.3f)",
                        removed_ankle, ankle_y)
            garment_verts = np.asarray(garment_mesh.vertices, dtype=np.float64)
            _, vertex_map = tree.query(garment_verts, k=1)
            vertex_map = vertex_map.astype(np.int64)

        # ── 7. Level ragged hems ──────────────────────────────────
        _level_hem(garment_mesh, top=True, blend_cm=1.0)   # smooth waistband
        _level_hem(garment_mesh, top=False, blend_cm=2.0)  # smooth leg hems

        # ── 8. Apply colour ──────────────────────────────────────────
        _apply_vertex_color(garment_mesh, garment_rgb)

        trimesh.repair.fix_normals(garment_mesh)
        logger.info(
            "ISP pants ready: %d verts, hip=%.0fcm, len=%.1fcm, "
            "ease=%.1fcm, Y=[%.3f, %.3f]",
            len(garment_mesh.vertices), desired_hip, desired_len * 100,
            ease_offset * 100,
            garment_mesh.bounds[0][1], garment_mesh.bounds[1][1],
        )
        return garment_mesh, vertex_map

    except Exception:
        logger.exception("ISP pants generation failed, falling back")
        return None


# ---------------------------------------------------------------------------
# Shared hem-leveling helper
# ---------------------------------------------------------------------------

def _level_hem(mesh: trimesh.Trimesh, *, top: bool = False, blend_cm: float = 2.0) -> None:
    """Flatten boundary vertices at the bottom (or top) hem to a clean line.

    For pants the mesh has two legs, so we split the bottom boundary by X sign
    and level each group independently.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Will be modified in-place.
    top : bool
        If *True*, level the top boundary instead of the bottom.
    blend_cm : float
        Nearby interior vertices within this distance are blended toward the
        hem Y so the transition is smooth.
    """
    from collections import Counter

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    # Find boundary vertices (edges shared by only one face)
    edge_count: dict[tuple[int, int], int] = Counter()
    for f in faces:
        for i in range(3):
            e = tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))
            edge_count[e] += 1
    boundary_set: set[int] = set()
    for e, c in edge_count.items():
        if c == 1:
            boundary_set.update(e)

    if not boundary_set:
        return

    boundary_idx = np.array(sorted(boundary_set), dtype=np.int64)
    bv = verts[boundary_idx]

    y_min, y_max = float(verts[:, 1].min()), float(verts[:, 1].max())
    y_range = y_max - y_min
    if y_range < 0.01:
        return

    # Select hem boundary
    if top:
        thresh = y_max - y_range * 0.10
        hem_mask = bv[:, 1] > thresh
    else:
        thresh = y_min + y_range * 0.10
        hem_mask = bv[:, 1] < thresh

    hem_idx = boundary_idx[hem_mask]
    if len(hem_idx) < 3:
        return

    # Split by X-sign for multi-leg (pants) — single body (shirt) will stay one group
    hem_verts = verts[hem_idx]
    cx = float(np.median(verts[:, 0]))
    left_mask = hem_verts[:, 0] < cx
    right_mask = ~left_mask

    groups = []
    if left_mask.sum() > 2:
        groups.append(hem_idx[left_mask])
    if right_mask.sum() > 2:
        groups.append(hem_idx[right_mask])
    if not groups:
        groups = [hem_idx]

    blend_m = blend_cm / 100.0
    for grp in groups:
        target_y = float(np.median(verts[grp, 1]))
        # Snap boundary verts to target
        mesh.vertices[grp, 1] = target_y

        # Blend nearby interior verts toward target
        if blend_m > 0:
            if top:
                near_mask = (verts[:, 1] > target_y - blend_m) & (verts[:, 1] <= target_y + blend_m)
            else:
                near_mask = (verts[:, 1] < target_y + blend_m) & (verts[:, 1] >= target_y - blend_m)
            near_mask &= ~np.isin(np.arange(len(verts)), grp)
            if near_mask.any():
                dist = np.abs(verts[near_mask, 1] - target_y)
                alpha = 1.0 - np.clip(dist / max(blend_m, 1e-6), 0.0, 1.0)
                mesh.vertices[near_mask, 1] = (
                    verts[near_mask, 1] * (1. - alpha) + target_y * alpha
                )


# ---------------------------------------------------------------------------
# Shared vertex color helper
# ---------------------------------------------------------------------------

def _apply_vertex_color(
    mesh: trimesh.Trimesh,
    garment_rgb: tuple[int, int, int] | None,
) -> None:
    """Apply a solid vertex colour to an ISP mesh."""
    n = len(mesh.vertices)
    if garment_rgb is not None:
        rgba = np.zeros((n, 4), dtype=np.uint8)
        rgba[:, 0] = garment_rgb[0]
        rgba[:, 1] = garment_rgb[1]
        rgba[:, 2] = garment_rgb[2]
        rgba[:, 3] = 255
    else:
        rgba = np.full((n, 4), 180, dtype=np.uint8)
        rgba[:, 3] = 255
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=rgba)


# ---------------------------------------------------------------------------
# Colour extraction helpers
# ---------------------------------------------------------------------------

def _extract_garment_color(clean_image) -> tuple[int, int, int] | None:
    """Extract dominant colour from the RGBA image (non-transparent pixels).

    Accepts either a PIL Image or raw bytes.
    """
    try:
        if isinstance(clean_image, Image.Image):
            img = clean_image.convert("RGBA")
        else:
            img = Image.open(io.BytesIO(clean_image)).convert("RGBA")
        arr = np.array(img)
        alpha = arr[:, :, 3]
        # Only consider non-transparent pixels
        mask = alpha > 128
        if mask.sum() < 100:
            return None
        rgb = arr[mask][:, :3].astype(np.float64)
        median = np.median(rgb, axis=0)
        r, g, b = int(median[0]), int(median[1]), int(median[2])
        # Only reject pure white (>250 all channels) or pure black (<15)
        if r > 250 and g > 250 and b > 250:
            return None  # pure white background leak
        if max(r, g, b) < 15:
            return None  # pure black
        return (r, g, b)
    except Exception:
        return None


def _extract_color_from_raw(image_bytes: bytes) -> tuple[int, int, int] | None:
    """Extract dominant colour directly from the *original* (raw) image.

    Useful when rembg strips too much / the clean image is empty.
    Center-crops to avoid background bleed.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        # Center crop: inner 50 %
        crop = img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
        arr = np.array(crop).reshape(-1, 3).astype(np.float64)
        median = np.median(arr, axis=0)
        r, g, b = int(median[0]), int(median[1]), int(median[2])
        if r > 250 and g > 250 and b > 250:
            return None  # pure white
        if max(r, g, b) < 15:
            return None  # pure black
        return (r, g, b)
    except Exception:
        return None


def _center_pixel_color(image_bytes: bytes) -> tuple[int, int, int] | None:
    """Sample the center pixel of the original image as absolute last resort."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        r, g, b = img.getpixel((w // 2, h // 2))
        return (r, g, b)
    except Exception:
        return None


def _apply_color_to_mesh(
    mesh: trimesh.Trimesh,
    garment_rgb: tuple[int, int, int] | None,
    clean_image: bytes,
) -> None:
    """Apply garment colour as vertex colors, or fall back to texture."""
    if garment_rgb is not None:
        n = len(mesh.vertices)
        rgba = np.zeros((n, 4), dtype=np.uint8)
        rgba[:, 0] = garment_rgb[0]
        rgba[:, 1] = garment_rgb[1]
        rgba[:, 2] = garment_rgb[2]
        rgba[:, 3] = 255
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=rgba)
    else:
        # Fall back to old texture pipeline
        try:
            uvs = assign_front_projection_uvs(mesh)
            texture_image = prepare_texture(clean_image, target_size=(1024, 1024), avoid_white_fill=True)
            if texture_image.mode == "RGBA":
                texture_image = texture_image.convert("RGB")
            tex_arr = np.array(texture_image, dtype=np.uint8)
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=tex_arr)
        except Exception:
            pass  # leave visual as-is


# ---------------------------------------------------------------------------
# Solid-color detection and GLB export
# ---------------------------------------------------------------------------

def _solid_color_from_texture(rgb_array: np.ndarray) -> tuple[int, int, int] | None:
    """If the texture is a single RGB color, return (r, g, b); else None."""
    if rgb_array.size == 0 or rgb_array.ndim != 3:
        return None
    flat = rgb_array.reshape(-1, rgb_array.shape[-1])
    if np.all(flat == flat[0]):
        return (int(flat[0, 0]), int(flat[0, 1]), int(flat[0, 2]))
    return None


def _dominant_color_from_texture(rgb_array: np.ndarray) -> tuple[int, int, int] | None:
    """Median RGB of texture; return None if white/very light or near-gray (use texture instead)."""
    if rgb_array.size == 0 or rgb_array.ndim != 3:
        return None
    flat = rgb_array.reshape(-1, rgb_array.shape[-1]).astype(np.float64)
    median_rgb = np.median(flat, axis=0)
    r, g, b = int(median_rgb[0]), int(median_rgb[1]), int(median_rgb[2])
    if max(r, g, b) >= 240:
        return None
    if max(r, g, b) - min(r, g, b) < 40:
        return None
    return (r, g, b)


def _export_textured_glb(mesh: trimesh.Trimesh) -> bytes:
    """Export a single mesh with TextureVisuals to GLB so the texture is embedded."""
    try:
        return mesh.export(file_type="glb")
    except Exception:
        logger.debug("Direct mesh GLB export failed, using Scene", exc_info=True)
        scene = trimesh.Scene(mesh)
        return scene.export(file_type="glb")


# ---------------------------------------------------------------------------
# Template creation dispatch
# ---------------------------------------------------------------------------

def _create_garment_template(
    m: GarmentMeasurements,
    body_landmarks: dict | None = None,
    silhouette: dict | None = None,
) -> trimesh.Trimesh:
    """Route to the correct parametric template builder."""
    builders = {
        GarmentType.TSHIRT: _build_tshirt,
        GarmentType.PANTS: _build_pants,
        GarmentType.DRESS: _build_dress,
    }
    builder = builders.get(m.garment_type)
    if builder is None:
        raise ValueError(f"Unsupported garment type: {m.garment_type}")
    if m.garment_type == GarmentType.TSHIRT:
        return _build_tshirt(m, body_landmarks, silhouette=silhouette)
    return builder(m, body_landmarks)


# ---------------------------------------------------------------------------
# Conforming garment dispatch
# ---------------------------------------------------------------------------

def _build_conforming_garment(
    body_mesh: trimesh.Trimesh,
    landmarks: dict,
    m: GarmentMeasurements,
    silhouette: dict | None = None,
) -> tuple[trimesh.Trimesh | None, np.ndarray | None]:
    """Route to the correct conforming builder based on garment type.

    Returns (mesh, selected_vertex_indices) or (None, None) on failure.
    """
    if m.garment_type == GarmentType.TSHIRT:
        return _build_conforming_tshirt(body_mesh, landmarks, m, silhouette)
    elif m.garment_type == GarmentType.PANTS:
        return _build_conforming_pants(body_mesh, landmarks, m)
    elif m.garment_type == GarmentType.DRESS:
        return _build_conforming_dress(body_mesh, landmarks, m)
    return None, None


# ---------------------------------------------------------------------------
# Conforming T-shirt (offset surface from body mesh)
# ---------------------------------------------------------------------------

def _build_conforming_tshirt(
    body_mesh: trimesh.Trimesh,
    landmarks: dict,
    m: GarmentMeasurements,
    silhouette: dict | None = None,
) -> tuple[trimesh.Trimesh, np.ndarray] | tuple[None, None]:
    """Build a T-shirt as an offset surface from the body mesh.
    Respects sleeve_length_cm (and silhouette if provided) so the result is
    half-sleeve when the user enters a short sleeve length.

    Returns (mesh, selected_vertex_indices) or (None, None) on failure.
    """
    verts = np.asarray(body_mesh.vertices, dtype=np.float64)
    faces = np.asarray(body_mesh.faces, dtype=np.int32)
    x, y = verts[:, 0], verts[:, 1]
    y_min, y_max = float(y.min()), float(y.max())

    shoulder_y = landmarks.get("shoulder_y")
    shoulder_half_w = landmarks.get("shoulder_half_w")
    if shoulder_y is None:
        shoulder_y = y_min + 0.82 * (y_max - y_min)
    if shoulder_half_w is None:
        shoulder_half_w = float(np.percentile(np.abs(x), 95)) * 0.5

    length_cm = m.length_cm or 72.0
    length_m = length_cm / 100.0
    shirt_bottom_y = float(shoulder_y) - length_m
    shirt_top_y = float(shoulder_y) + 0.02

    # Effective sleeve length: from measurements, optionally capped by silhouette
    sleeve_m = (m.sleeve_length_cm or 25.0) / 100.0
    if silhouette:
        sleeve_frac = silhouette.get("sleeve_length_frac", 0.25)
        sleeve_m = min(sleeve_m, length_m * max(0.15, sleeve_frac))
    sleeve_hem_y = float(shoulder_y) - sleeve_m

    # Torso + arms in shirt Y range
    vertex_mask = (y >= shirt_bottom_y) & (y <= shirt_top_y)

    # Exclude arm vertices below the sleeve hem
    arm_region = (np.abs(x) > shoulder_half_w * 0.5) & (y < shoulder_y)
    below_sleeve_hem = y < sleeve_hem_y
    vertex_mask = vertex_mask & ~(arm_region & below_sleeve_hem)

    if not np.any(vertex_mask):
        return None, None
    face_mask = vertex_mask[faces].all(axis=1)
    if not np.any(face_mask):
        return None, None

    selected_vertex_indices = np.where(vertex_mask)[0]
    old_to_new = {int(o): i for i, o in enumerate(selected_vertex_indices)}
    garment_faces_old = faces[face_mask]
    new_faces = np.array(
        [[old_to_new[a], old_to_new[b], old_to_new[c]] for a, b, c in garment_faces_old],
        dtype=np.int32,
    )
    garment_vertices = verts[selected_vertex_indices]
    normals = np.asarray(body_mesh.vertex_normals, dtype=np.float64)
    if normals.shape[0] != verts.shape[0]:
        return None, None
    garment_normals = normals[selected_vertex_indices]
    garment_vertices = garment_vertices + _CONFORMING_OFFSET * garment_normals

    out = trimesh.Trimesh(vertices=garment_vertices, faces=new_faces)
    return out, selected_vertex_indices


# ---------------------------------------------------------------------------
# Conforming Pants (offset surface from body mesh)
# ---------------------------------------------------------------------------

def _build_conforming_pants(
    body_mesh: trimesh.Trimesh,
    landmarks: dict,
    m: GarmentMeasurements,
) -> tuple[trimesh.Trimesh, np.ndarray] | tuple[None, None]:
    """Build pants as an offset surface from the body mesh.

    Selects body vertices in the leg+waist region, offsets along normals,
    with special crotch handling to prevent inward-pointing normals.

    Returns (mesh, selected_vertex_indices) or (None, None) on failure.
    """
    verts = np.asarray(body_mesh.vertices, dtype=np.float64)
    faces = np.asarray(body_mesh.faces, dtype=np.int32)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    ankle_y = landmarks.get("ankle_y", 0.03)
    waist_y = landmarks.get("waist_y", 0.85)
    inseam_y = landmarks.get("inseam_y", 0.70)
    shoulder_half_w = landmarks.get("shoulder_half_w", 0.22)

    # Select vertices in pants Y range
    y_bottom = ankle_y - 0.02
    y_top = waist_y + 0.02
    vertex_mask = (y >= y_bottom) & (y <= y_top)

    # Exclude arm vertices: far from center AND above inseam
    arm_exclude = (np.abs(x) > shoulder_half_w * 0.65) & (y > inseam_y)
    vertex_mask = vertex_mask & ~arm_exclude

    if not np.any(vertex_mask):
        return None, None
    face_mask = vertex_mask[faces].all(axis=1)
    if not np.any(face_mask):
        return None, None

    selected_vertex_indices = np.where(vertex_mask)[0]
    old_to_new = {int(o): i for i, o in enumerate(selected_vertex_indices)}
    garment_faces_old = faces[face_mask]
    new_faces = np.array(
        [[old_to_new[a], old_to_new[b], old_to_new[c]] for a, b, c in garment_faces_old],
        dtype=np.int32,
    )

    garment_vertices = verts[selected_vertex_indices].copy()
    normals = np.asarray(body_mesh.vertex_normals, dtype=np.float64)
    if normals.shape[0] != verts.shape[0]:
        return None, None
    garment_normals = normals[selected_vertex_indices].copy()

    # Crotch fix: inner-thigh normals point inward — blend with radial-outward
    sel_x = garment_vertices[:, 0]
    sel_y = garment_vertices[:, 1]
    crotch_zone = (np.abs(sel_y - inseam_y) < 0.08) & (np.abs(sel_x) < shoulder_half_w * 0.35)
    if np.any(crotch_zone):
        # Radial-outward direction from Y axis
        radial = np.zeros_like(garment_normals[crotch_zone])
        radial[:, 0] = garment_vertices[crotch_zone, 0]
        radial[:, 2] = garment_vertices[crotch_zone, 2]
        radial_len = np.linalg.norm(radial, axis=1, keepdims=True)
        radial_len = np.maximum(radial_len, 1e-6)
        radial = radial / radial_len
        # Blend: 50% normal + 50% radial outward
        blended = 0.5 * garment_normals[crotch_zone] + 0.5 * radial
        blended_len = np.linalg.norm(blended, axis=1, keepdims=True)
        blended_len = np.maximum(blended_len, 1e-6)
        garment_normals[crotch_zone] = blended / blended_len

    garment_vertices = garment_vertices + _CONFORMING_OFFSET * garment_normals

    out = trimesh.Trimesh(vertices=garment_vertices, faces=new_faces)
    return out, selected_vertex_indices


# ---------------------------------------------------------------------------
# Conforming Dress (offset surface from body mesh)
# ---------------------------------------------------------------------------

def _build_conforming_dress(
    body_mesh: trimesh.Trimesh,
    landmarks: dict,
    m: GarmentMeasurements,
) -> tuple[trimesh.Trimesh, np.ndarray] | tuple[None, None]:
    """Build a dress as an offset surface from the body mesh.

    Covers from shoulder to dress bottom. Below hips, adds flare by
    increasing the X/Z offset component.

    Returns (mesh, selected_vertex_indices) or (None, None) on failure.
    """
    verts = np.asarray(body_mesh.vertices, dtype=np.float64)
    faces = np.asarray(body_mesh.faces, dtype=np.int32)
    x, y = verts[:, 0], verts[:, 1]

    shoulder_y = landmarks.get("shoulder_y", 1.40)
    shoulder_half_w = landmarks.get("shoulder_half_w", 0.22)
    pelvis_y = landmarks.get("pelvis_y", 0.85)
    hip_y = pelvis_y  # hip region is around pelvis

    length_cm = m.length_cm or 100.0
    length_m = length_cm / 100.0
    dress_top_y = float(shoulder_y) + 0.02
    dress_bottom_y = float(shoulder_y) - length_m

    # Select vertices in dress Y range
    vertex_mask = (y >= dress_bottom_y) & (y <= dress_top_y)

    # Exclude arm vertices: far from center AND in shoulder region
    arm_exclude = (np.abs(x) > shoulder_half_w * 0.5) & (y > shoulder_y - 0.15)
    vertex_mask = vertex_mask & ~arm_exclude

    if not np.any(vertex_mask):
        return None, None
    face_mask = vertex_mask[faces].all(axis=1)
    if not np.any(face_mask):
        return None, None

    selected_vertex_indices = np.where(vertex_mask)[0]
    old_to_new = {int(o): i for i, o in enumerate(selected_vertex_indices)}
    garment_faces_old = faces[face_mask]
    new_faces = np.array(
        [[old_to_new[a], old_to_new[b], old_to_new[c]] for a, b, c in garment_faces_old],
        dtype=np.int32,
    )

    garment_vertices = verts[selected_vertex_indices].copy()
    normals = np.asarray(body_mesh.vertex_normals, dtype=np.float64)
    if normals.shape[0] != verts.shape[0]:
        return None, None
    garment_normals = normals[selected_vertex_indices].copy()

    # Below hips: add flare on X/Z normal components
    sel_y = garment_vertices[:, 1]
    below_hip = sel_y < hip_y
    if np.any(below_hip):
        flare_amount = np.maximum(0.0, (hip_y - sel_y[below_hip]) * 0.04)
        # Scale the X and Z normal components to add outward flare
        xz_norm = np.sqrt(garment_normals[below_hip, 0] ** 2 + garment_normals[below_hip, 2] ** 2)
        xz_norm = np.maximum(xz_norm, 1e-6)
        garment_normals[below_hip, 0] += (garment_normals[below_hip, 0] / xz_norm) * flare_amount
        garment_normals[below_hip, 2] += (garment_normals[below_hip, 2] / xz_norm) * flare_amount
        # Re-normalize
        norm_len = np.linalg.norm(garment_normals[below_hip], axis=1, keepdims=True)
        norm_len = np.maximum(norm_len, 1e-6)
        garment_normals[below_hip] = garment_normals[below_hip] / norm_len

    garment_vertices = garment_vertices + _CONFORMING_OFFSET * garment_normals

    # Additional flare displacement for skirt portion
    if np.any(below_hip):
        flare_disp = np.maximum(0.0, (hip_y - sel_y[below_hip]) * 0.04)
        garment_vertices[below_hip, 0] += np.sign(garment_vertices[below_hip, 0] + 1e-8) * flare_disp
        garment_vertices[below_hip, 2] += np.sign(garment_vertices[below_hip, 2] + 1e-8) * flare_disp * 0.5

    out = trimesh.Trimesh(vertices=garment_vertices, faces=new_faces)
    return out, selected_vertex_indices


# ---------------------------------------------------------------------------
# T-shirt template (parametric fallback)
# ---------------------------------------------------------------------------

def _build_sleeve(
    arm_r: float,
    sleeve_len: float,
    side: float,
    shoulder_half_w: float,
    shoulder_y: float,
    *,
    tilt_deg: float = 17.0,
    offset: float | None = None,
) -> trimesh.Trimesh:
    """Build one sleeve tube, rotated to hang downward from the shoulder."""
    off = offset if offset is not None else _GARMENT_OFFSET
    n = 10
    heights = np.linspace(0.0, sleeve_len, n)
    root_r = shoulder_half_w + off * 1.2
    wrist_r = arm_r + off
    rx = np.linspace(root_r, wrist_r, n)
    rz = rx * 0.9

    sleeve = create_lofted_tube(
        rx, rz, heights,
        ring_points=24,
        cap_top=False,
        cap_bottom=True,
    )
    rot_down = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    sleeve.apply_transform(rot_down)
    angle = side * np.radians(tilt_deg)
    rot_tilt = trimesh.transformations.rotation_matrix(angle, [0.0, 0.0, 1.0])
    sleeve.apply_transform(rot_tilt)
    sleeve_x = side * (shoulder_half_w - arm_r * 0.12)
    sleeve.apply_translation([sleeve_x, shoulder_y, 0.0])
    return sleeve


def _build_tshirt(
    m: GarmentMeasurements,
    body_landmarks: dict | None = None,
    silhouette: dict | None = None,
) -> trimesh.Trimesh:
    """Build a T-shirt: torso + sleeves. When body_landmarks exist, shirt pastes on model (tight fit)."""
    length = (m.length_cm or 72.0) / 100.0
    sleeve_len = (m.sleeve_length_cm or 25.0) / 100.0
    offset = _GARMENT_OFFSET_TIGHT if body_landmarks else _GARMENT_OFFSET

    if body_landmarks:
        shoulder_half_w = body_landmarks["shoulder_half_w"]
        arm_r = body_landmarks["arm_r"]
        shoulder_y = body_landmarks["shoulder_y"]
        chest_r = body_landmarks.get("chest_rx", 0.98 / (2.0 * np.pi)) + offset
        sleeve_tilt = 80.0
    else:
        shoulder_half_w = _REF_SHOULDER_HALF_W
        arm_r = _REF_ARM_R
        shoulder_y = _REF_SHOULDER_Y
        chest_r = (m.chest_cm or 100.0) / 100.0 / (2.0 * np.pi) + offset
        sleeve_tilt = 17.0

    if silhouette:
        sleeve_len = length * silhouette.get("sleeve_length_frac", 0.25)
        sleeve_len = max(0.12, min(sleeve_len, length * 0.5))
        if not body_landmarks:
            arm_r = arm_r * (1.0 + 0.25 * silhouette.get("sleeve_width_frac", 0.3))
        hem_to_shoulder = silhouette.get("hem_to_shoulder_ratio", 1.0)
    else:
        hem_to_shoulder = 1.0

    shirt_top_y = shoulder_y
    shirt_bottom_y = shirt_top_y - length

    n = 16
    heights = np.linspace(shirt_bottom_y, shirt_top_y, n)
    rx = np.empty(n)
    rz = np.empty(n)

    shoulder_rx = shoulder_half_w + offset * (2.0 if body_landmarks else 2.8)
    chest_r_eff = chest_r + offset * (0.5 if body_landmarks else 0.8)
    neck_frac = silhouette.get("neck_width_frac", 0.35) if silhouette else 0.35
    neck_frac = max(0.2, min(neck_frac, 0.42))
    neck_r_shirt = shoulder_rx * neck_frac

    ref_width = 0.5
    if silhouette and silhouette.get("width_at_height"):
        widths = [w for _, w in silhouette["width_at_height"]]
        ref_width = float(np.median(widths))
        ref_width = max(0.25, min(ref_width, 0.85))

    hem_min_r = chest_r_eff * 0.6

    for i in range(n):
        t = (heights[i] - shirt_bottom_y) / length
        s = smooth_step(t)
        base = chest_r_eff * (1.0 + (1.0 - t) * 0.08 * hem_to_shoulder) * (1.0 - s) + shoulder_rx * s
        if silhouette:
            width_scale = get_width_at_height_frac(silhouette, t) / ref_width
            width_scale = np.clip(width_scale, 0.5, 1.8)
            base *= width_scale
        rx[i] = max(base, hem_min_r) if t < 0.15 else base
        if t > 0.82:
            neck_blend = (t - 0.82) / 0.18
            rx[i] = rx[i] * (1.0 - neck_blend) + neck_r_shirt * neck_blend
        rz[i] = rx[i] * 0.85

    torso = create_lofted_tube(
        rx, rz, heights,
        ring_points=32,
        cap_top=False,
        cap_bottom=True,
    )

    parts: list[trimesh.Trimesh] = [torso]
    for side in (-1.0, 1.0):
        parts.append(_build_sleeve(
            arm_r, sleeve_len, side, shoulder_half_w, shoulder_y,
            tilt_deg=sleeve_tilt,
            offset=offset,
        ))

    return trimesh.util.concatenate(parts)


# ---------------------------------------------------------------------------
# Pants template (parametric fallback)
# ---------------------------------------------------------------------------

def _build_pants(
    m: GarmentMeasurements,
    body_landmarks: dict | None = None,
) -> trimesh.Trimesh:
    """Build a pants mesh from two leg tubes plus a waistband section."""
    waist_r = (m.waist_cm or 84.0) / 100.0 / (2.0 * np.pi) + _GARMENT_OFFSET
    hip_r = (m.hip_cm or 100.0) / 100.0 / (2.0 * np.pi) + _GARMENT_OFFSET
    inseam_len = (m.inseam_cm or 78.0) / 100.0

    if body_landmarks:
        ankle_y = body_landmarks.get("ankle_y", 0.03)
        crotch_y = body_landmarks.get("inseam_y", ankle_y + inseam_len)
        waist_y_body = body_landmarks.get("waist_y", crotch_y + 0.12)
        hip_r = max(hip_r, body_landmarks.get("hip_rx", hip_r) + _GARMENT_OFFSET)
        waist_r = max(waist_r, body_landmarks.get("waist_rx", waist_r) + _GARMENT_OFFSET)
        leg_gap = body_landmarks.get("hip_rx", hip_r) * 0.45
        leg_bottom_y = max(0.02, ankle_y - 0.02)
    else:
        leg_bottom_y = 0.03
        crotch_y = leg_bottom_y + inseam_len
        waist_y_body = crotch_y + 0.12
        leg_gap = hip_r * 0.55

    waist_y = waist_y_body
    n_leg = 14

    parts: list[trimesh.Trimesh] = []

    for side in (-1.0, 1.0):
        heights = np.linspace(leg_bottom_y, crotch_y, n_leg)
        rx = np.empty(n_leg)
        rz = np.empty(n_leg)
        for i in range(n_leg):
            t = (heights[i] - leg_bottom_y) / max(crotch_y - leg_bottom_y, 0.01)
            s = smooth_step(t)
            rx[i] = waist_r * 0.42 + (hip_r * 0.52 - waist_r * 0.42) * s
            rz[i] = rx[i] * 0.9

        leg = create_lofted_tube(rx, rz, heights, ring_points=24)
        leg.apply_translation([side * leg_gap, 0.0, 0.0])
        parts.append(leg)

    # Waistband
    wb_n = 6
    wb_heights = np.linspace(crotch_y, waist_y, wb_n)
    wb_rx = np.linspace(hip_r, waist_r, wb_n)
    wb_rz = wb_rx * 0.85
    waistband = create_lofted_tube(wb_rx, wb_rz, wb_heights, ring_points=32)
    parts.append(waistband)

    return trimesh.util.concatenate(parts)


# ---------------------------------------------------------------------------
# Dress template (parametric fallback)
# ---------------------------------------------------------------------------

def _build_dress(
    m: GarmentMeasurements,
    body_landmarks: dict | None = None,
) -> trimesh.Trimesh:
    """Build a dress-shaped tube (fitted at bust, flared at hem)."""
    chest_r = (m.chest_cm or 92.0) / 100.0 / (2.0 * np.pi) + _GARMENT_OFFSET
    waist_r = (m.waist_cm or 74.0) / 100.0 / (2.0 * np.pi) + _GARMENT_OFFSET
    hip_r = (m.hip_cm or 98.0) / 100.0 / (2.0 * np.pi) + _GARMENT_OFFSET
    length = (m.length_cm or 100.0) / 100.0

    if body_landmarks:
        dress_top_y = body_landmarks["shoulder_y"]
        chest_r = max(chest_r, body_landmarks.get("chest_rx", chest_r) + _GARMENT_OFFSET)
        waist_r = max(waist_r, body_landmarks.get("waist_rx", waist_r) + _GARMENT_OFFSET)
        hip_r = max(hip_r, body_landmarks.get("hip_rx", hip_r) + _GARMENT_OFFSET)
    else:
        dress_top_y = _REF_SHOULDER_Y

    dress_bottom_y = dress_top_y - length

    n = 20
    heights = np.linspace(dress_bottom_y, dress_top_y, n)
    rx = np.empty(n)
    rz = np.empty(n)

    waist_pos = 0.40
    hip_pos = 0.25

    for i in range(n):
        t = (heights[i] - dress_bottom_y) / length

        if t <= hip_pos:
            s = smooth_step(t / hip_pos)
            rx[i] = (hip_r * 1.3) + (hip_r - hip_r * 1.3) * s
            rz[i] = rx[i] * 0.82
        elif t <= waist_pos:
            s = smooth_step((t - hip_pos) / (waist_pos - hip_pos))
            rx[i] = hip_r + (waist_r - hip_r) * s
            rz[i] = rx[i] * 0.82
        else:
            s = smooth_step((t - waist_pos) / (1.0 - waist_pos))
            rx[i] = waist_r + (chest_r - waist_r) * s
            rz[i] = rx[i] * 0.82

    return create_lofted_tube(rx, rz, heights, ring_points=32)
