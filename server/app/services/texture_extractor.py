"""Texture Extractor Service.

Handles background removal from uploaded garment images and
prepares the cleaned image for use as a 3D texture map.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from rembg import remove

# Max dimension for rembg input to avoid ONNX "bad allocation" on large images
_REMBG_MAX_PIXELS = 1024


def remove_background(
    image_bytes: bytes,
    garment_type: str | None = None,
) -> Image.Image:
    """Remove the background from a garment photograph.

    Uses default rembg (u2net) so the full garment is kept as foreground.
    This works best for flat-lay photos (shirt on white background). Cloth-aware
    segmentation (u2net_cloth_seg) is not used here because it is trained on
    people wearing clothes and often produces a bad or tiny mask on flat-lay
    images, which would break silhouette analysis and the parametric shape.

    Input is resized so the longest side is at most _REMBG_MAX_PIXELS before
    running the model to avoid ONNX Runtime "bad allocation" on large images.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image file (JPEG, PNG, or WebP).
    garment_type : str, optional
        Unused; kept for API compatibility. Silhouette uses the full mask.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > _REMBG_MAX_PIXELS:
        scale = _REMBG_MAX_PIXELS / max(w, h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    to_remove = buf.getvalue()
    output_bytes = remove(to_remove)
    return Image.open(io.BytesIO(output_bytes)).convert("RGBA")


def _is_mostly_solid_color(image: Image.Image, threshold_std: float = 48.0) -> bool:
    """True if opaque pixels have low color variance (e.g. solid red shirt after rembg)."""
    arr = np.array(image)
    opaque = arr[:, :, 3] > 200
    if opaque.sum() < 100:
        opaque = arr[:, :, 3] > 128
    if opaque.sum() < 50:
        return False
    rgb = arr[opaque][:, :3].astype(np.float64)
    return float(np.std(rgb)) < threshold_std


def prepare_texture(
    image: Image.Image,
    target_size: tuple[int, int] = (1024, 1024),
    avoid_white_fill: bool = False,
) -> Image.Image:
    """Centre the garment image on a square texture canvas.

    The garment is scaled to fill *target_size* (cover) so it occupies most of
    the texture and the uploaded color/print is visible. Padding uses the
    dominant garment color so any UV bleed matches the garment.
    For nearly solid-color images (e.g. plain red shirt), the entire texture
    is filled with that color so the 3D garment shows the correct color
    regardless of UV mapping.
    """
    # use_center_region when avoiding white so solid dark colors (grey, black) aren't pulled to white by rembg edges
    fill_rgb = extract_dominant_color(image, avoid_white=avoid_white_fill, use_center_region=avoid_white_fill)
    # Never use white as solid fill when we have a colored garment (rembg can leave white dominating)
    if avoid_white_fill and all(c >= 230 for c in fill_rgb):
        fill_rgb = extract_dominant_color(image, avoid_white=True, use_center_region=True)

    # Solid-color garment: fill whole texture so color is correct everywhere (and backend will use vertex colors)
    if _is_mostly_solid_color(image, threshold_std=55.0):
        return Image.new("RGB", target_size, fill_rgb)
    # If dominant color is strongly saturated (e.g. clear red / dark green), use slightly looser solid check
    if _is_mostly_solid_color(image, threshold_std=68.0) and max(abs(c - 128) for c in fill_rgb) > 30:
        return Image.new("RGB", target_size, fill_rgb)
    # When avoiding white and we have a clear non-white dominant color with mostly solid image, force solid fill so vertex colors are used
    if avoid_white_fill and any(c < 230 for c in fill_rgb) and _is_mostly_solid_color(image, threshold_std=72.0):
        return Image.new("RGB", target_size, fill_rgb)

    img_copy = image.copy().convert("RGBA")
    w, h = target_size[0], target_size[1]
    iw, ih = img_copy.size

    # Scale to fill then center-crop so the garment stays centered; front-facing UV band samples this center
    scale = max(w / iw, h / ih)
    new_w = max(1, int(iw * scale))
    new_h = max(1, int(ih * scale))
    img_copy = img_copy.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x0 = (new_w - w) // 2
    y0 = (new_h - h) // 2
    img_copy = img_copy.crop((x0, y0, x0 + w, y0 + h))

    canvas = Image.new("RGBA", target_size, (*fill_rgb, 255))
    canvas.paste(img_copy, (0, 0), img_copy)

    return canvas


def _crop_to_upper_body(
    image: Image.Image,
    upper_height_frac: float = 0.55,
    torso_band: tuple[float, float] | None = None,
) -> Image.Image:
    """Crop to the upper portion of the opaque region (torso/shirt) for person photos.

    Uses alpha to find the bounding box. If torso_band is (low, high) e.g. (0.2, 0.65),
    keeps only that vertical band (20%–65% from top) so shirt dominates over face/legs.
    Otherwise keeps the top upper_height_frac.
    """
    arr = np.array(image)
    if arr.ndim < 3:
        return image
    alpha = arr[:, :, 3] if arr.shape[2] >= 4 else np.ones((arr.shape[0], arr.shape[1]), dtype=np.uint8) * 255
    opaque = alpha >= 128
    if not np.any(opaque):
        return image
    rows = np.any(opaque, axis=1)
    cols = np.any(opaque, axis=0)
    r_min, r_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    c_min, c_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    h = r_max - r_min + 1
    if torso_band is not None:
        t_low, t_high = torso_band
        r_crop_min = r_min + int(h * t_low)
        r_crop_max = r_min + int(h * t_high)
        r_crop_max = min(r_crop_max, r_max)
        return image.crop((c_min, r_crop_min, c_max + 1, r_crop_max + 1))
    r_crop = r_min + int(h * upper_height_frac)
    r_crop = min(r_crop, r_max - 1)
    return image.crop((c_min, r_min, c_max + 1, r_crop + 1))


def prepare_texture_for_person_photo(
    image: Image.Image,
    target_size: tuple[int, int] = (1024, 1024),
    upper_height_frac: float = 0.55,
    torso_band: tuple[float, float] = (0.15, 0.60),
) -> Image.Image:
    """Like prepare_texture but crops to torso band first so shirt color/texture dominate (not face/white)."""
    cropped = _crop_to_upper_body(image, upper_height_frac=upper_height_frac, torso_band=torso_band)
    return prepare_texture(cropped, target_size=target_size, avoid_white_fill=True)


def _dominant_from_rgb(rgb: np.ndarray, avoid_white: bool = False) -> tuple[int, int, int]:
    """Compute median RGB; if avoid_white and result is white-ish, exclude bright pixels and retry."""
    if rgb.size == 0:
        return (128, 128, 128)
    # When avoiding white, exclude bright pixels first so background/rembg edges don't dominate
    if avoid_white:
        bright = np.max(rgb, axis=1)
        non_white = bright < 220
        if non_white.sum() >= 50:
            rgb = rgb[non_white]
    if rgb.size == 0:
        return (128, 128, 128)
    median_rgb = np.median(rgb, axis=0).astype(int)
    out = (int(median_rgb[0]), int(median_rgb[1]), int(median_rgb[2]))
    # If still white-ish, retry with stricter threshold
    if avoid_white and all(c > 200 for c in out):
        non_white = np.any(rgb < 200, axis=1)
        if non_white.sum() >= 50:
            rgb = rgb[non_white]
            median_rgb = np.median(rgb, axis=0).astype(int)
            out = (int(median_rgb[0]), int(median_rgb[1]), int(median_rgb[2]))
    return out


def extract_dominant_color(
    image: Image.Image,
    avoid_white: bool = False,
    use_center_region: bool = False,
) -> tuple[int, int, int]:
    """Return a representative non-transparent colour of *image*.

    Uses median of bright opaque pixels to avoid darkening from shadows
    and edge artifacts. If avoid_white is True (e.g. for person photos),
    avoids returning white/light gray so the garment color shows.
    If use_center_region is True, samples only from the inner 60% of the
    garment bbox so rembg edge artifacts (light/grey fringing) don't pull
    the dominant color toward white.
    """
    arr = np.array(image)
    h, w = arr.shape[:2]

    # Mask: only pixels that are highly opaque (exclude edge artifacts)
    opaque = arr[:, :, 3] > 200
    if not opaque.any():
        opaque = arr[:, :, 3] > 128
    if not opaque.any():
        return (128, 128, 128)

    if use_center_region:
        # Restrict to one garment so we don't sample the gap in front+back or multi-panel images
        rows = np.any(opaque, axis=1)
        cols = np.any(opaque, axis=0)
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        bbox_w = c_max - c_min + 1
        bbox_h = r_max - r_min + 1
        r_margin = int(0.2 * max(1, bbox_h))
        c_margin = int(0.2 * max(1, bbox_w))
        r_lo, r_hi = r_min + r_margin, r_max - r_margin
        c_lo, c_hi = c_min + c_margin, c_max - c_margin
        # If image is wide (e.g. front+back side by side), sample from left half only so we get one garment's color
        if w > 1.35 * h and bbox_w > bbox_h:
            c_hi = c_min + (c_max - c_min) // 2
        if r_hi > r_lo and c_hi > c_lo:
            center_mask = np.zeros_like(opaque, dtype=bool)
            center_mask[r_lo:r_hi + 1, c_lo:c_hi + 1] = True
            opaque = opaque & center_mask
    if not opaque.any():
        opaque = arr[:, :, 3] > 128

    rgb = arr[opaque][:, :3].astype(np.float64)
    if rgb.size == 0:
        return (128, 128, 128)
    # Exclude very dark pixels (shadows)
    brightness = np.max(rgb, axis=1)
    bright_mask = brightness >= 40
    if bright_mask.sum() >= 100:
        rgb = rgb[bright_mask]

    return _dominant_from_rgb(rgb, avoid_white=avoid_white)
