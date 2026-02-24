"""Garment silhouette analysis from a segmented (background-removed) image.

Extracts width-at-height profile, neckline, and sleeve proportions
so the 3D template can match the uploaded garment's shape.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from PIL import Image


class TShirtSilhouette(TypedDict):
    """Shape cues from a t-shirt image (normalized 0–1 where relevant)."""
    # Width of the garment as a fraction of image width, at each height fraction (0=bottom, 1=top)
    width_at_height: list[tuple[float, float]]
    # Neckline width as fraction of torso width at shoulder
    neck_width_frac: float
    # Sleeve length as fraction of total image height (how far down the sleeve extends)
    sleeve_length_frac: float
    # Sleeve width (at armhole) as fraction of image width
    sleeve_width_frac: float
    # Relaxed/oversized: hem width vs shoulder width ratio (>1 = A-line or boxy)
    hem_to_shoulder_ratio: float
    # Image aspect (width/height) for texture mapping
    aspect: float


def _opaque_mask(arr: np.ndarray, alpha_thresh: int = 128) -> np.ndarray:
    """Boolean mask of opaque pixels."""
    return arr[:, :, 3] >= alpha_thresh


def analyze_tshirt_silhouette(image: Image.Image) -> TShirtSilhouette:
    """Analyze a segmented t-shirt image to extract silhouette proportions.

    Assumes the image is roughly vertical (neck at top, hem at bottom).
    For wide images (e.g. front+back side by side), analyzes left half so width/sleeve come from a single garment.
    Returns normalized proportions so the 3D template can be scaled to match.
    """
    arr = np.array(image)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return _default_tshirt_silhouette()

    # Single-garment region: when image is wide, run analysis on left half
    if w > 1.35 * h:
        half_w = w // 2
        arr = arr[:, :half_w].copy()
        w = half_w
    mask = _opaque_mask(arr)
    if not mask.any():
        return _default_tshirt_silhouette()

    # Bounding box of the garment
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    bbox_h = r_max - r_min + 1
    bbox_w = c_max - c_min + 1

    # Sample at N heights (0 = bottom of bbox, 1 = top)
    n_slices = 12
    width_at_height: list[tuple[float, float]] = []

    for i in range(n_slices):
        # y_frac: 0 = hem (bottom), 1 = neck (top)
        y_frac = (n_slices - 1 - i) / max(n_slices - 1, 1)
        row = r_min + int(y_frac * (bbox_h - 1))
        row = np.clip(row, 0, h - 1)
        strip = mask[row, :]
        if strip.any():
            left = np.argmax(strip)
            right = len(strip) - 1 - np.argmax(strip[::-1])
            width_px = max(1, right - left + 1)
            width_frac = width_px / w
        else:
            width_frac = 0.15
        width_at_height.append((y_frac, width_frac))

    # Neck: width at top 15% of bbox
    neck_slice = max(0, int(0.85 * (bbox_h - 1)))
    neck_row = r_min + neck_slice
    neck_row = np.clip(neck_row, 0, h - 1)
    neck_strip = mask[neck_row, :]
    if neck_strip.any():
        nl = np.argmax(neck_strip)
        nr = len(neck_strip) - 1 - np.argmax(neck_strip[::-1])
        neck_width_px = max(1, nr - nl + 1)
        neck_width_frac = neck_width_px / w
        # Torso width at "shoulder" (around 80% height)
        shoulder_slice = max(0, int(0.75 * (bbox_h - 1)))
        sh_row = r_min + shoulder_slice
        sh_row = np.clip(sh_row, 0, h - 1)
        sh_strip = mask[sh_row, :]
        if sh_strip.any():
            sl = np.argmax(sh_strip)
            sr = len(sh_strip) - 1 - np.argmax(sh_strip[::-1])
            shoulder_width_px = max(1, sr - sl + 1)
            neck_width_frac = min(0.95, neck_width_px / max(shoulder_width_px, 1))
    else:
        neck_width_frac = 0.35

    # Hem vs shoulder: width at bottom vs width at ~75% height
    hem_width = width_at_height[0][1] if width_at_height else 0.5
    shoulder_width = width_at_height[-1][1] if width_at_height else 0.5
    for _, wf in width_at_height:
        if wf > shoulder_width:
            shoulder_width = wf
    hem_to_shoulder = hem_width / max(shoulder_width, 0.01)

    # Sleeve length from image: find where the "wide" shoulder/sleeve region narrows (sleeve hem)
    step = max(1, bbox_h // 30)
    widths_by_row: list[tuple[int, float]] = []
    for row in range(r_min, r_max + 1, step):
        strip = mask[row, :]
        if strip.any():
            left = np.argmax(strip)
            right = len(strip) - 1 - np.argmax(strip[::-1])
            widths_by_row.append((row, (right - left + 1) / w))
    if widths_by_row:
        rows_sorted = sorted(widths_by_row, key=lambda x: x[0])
        upper_frac = 0.35
        n_upper = max(1, int(len(rows_sorted) * upper_frac))
        max_upper_width = max(ww for _, ww in rows_sorted[:n_upper])
        # Lowered from 0.85 → 0.75 so half-sleeve narrowing is detected
        threshold = 0.75 * max_upper_width
        in_wide_region = False
        sleeve_hem_row = r_min + bbox_h // 2  # fallback

        # --- Primary: threshold-crossing approach ---
        found_via_threshold = False
        for row_val, width_val in rows_sorted:
            if width_val >= threshold:
                in_wide_region = True
            elif in_wide_region:
                sleeve_hem_row = row_val
                found_via_threshold = True
                break

        # --- Fallback: derivative-based (sharpest width drop) ---
        if not found_via_threshold and len(rows_sorted) >= 4:
            widths_arr = np.array([ww for _, ww in rows_sorted])
            rows_arr = np.array([rr for rr, _ in rows_sorted])
            grad = np.diff(widths_arr)
            # Only consider the upper 60 % of the garment (shoulder/sleeve area)
            upper_limit = max(1, int(len(grad) * 0.6))
            grad_upper = grad[:upper_limit]
            if len(grad_upper) > 0:
                sharpest_idx = int(np.argmin(grad_upper))  # largest negative drop
                if grad_upper[sharpest_idx] < -0.02:  # meaningful narrowing
                    sleeve_hem_row = int(rows_arr[sharpest_idx + 1])

        sleeve_length_frac = (sleeve_hem_row - r_min) / max(bbox_h, 1)
        sleeve_length_frac = min(0.45, max(0.15, sleeve_length_frac))
    else:
        sleeve_length_frac = 0.25

    # Sleeve width: at armhole level, width is large; use shoulder-level width as proxy
    sleeve_width_frac = min(0.5, shoulder_width * 1.1)

    return TShirtSilhouette(
        width_at_height=width_at_height,
        neck_width_frac=float(np.clip(neck_width_frac, 0.2, 0.6)),
        sleeve_length_frac=float(np.clip(sleeve_length_frac, 0.12, 0.5)),
        sleeve_width_frac=float(np.clip(sleeve_width_frac, 0.2, 0.6)),
        hem_to_shoulder_ratio=float(np.clip(hem_to_shoulder, 0.7, 1.5)),
        aspect=w / max(h, 1),
    )


def _default_tshirt_silhouette() -> TShirtSilhouette:
    """Fallback when image cannot be analyzed."""
    return TShirtSilhouette(
        width_at_height=[(0.0, 0.5), (1.0, 0.5)],
        neck_width_frac=0.35,
        sleeve_length_frac=0.25,
        sleeve_width_frac=0.3,
        hem_to_shoulder_ratio=1.0,
        aspect=1.0,
    )


def get_width_at_height_frac(silhouette: TShirtSilhouette, height_frac: float) -> float:
    """Interpolate width fraction at a given height (0=hem, 1=shoulder/neck)."""
    profile = silhouette["width_at_height"]
    if not profile:
        return 0.5
    heights = np.array([p[0] for p in profile])
    widths = np.array([p[1] for p in profile])
    return float(np.interp(height_frac, heights, widths))
