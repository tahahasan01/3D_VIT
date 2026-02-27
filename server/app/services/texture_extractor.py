"""Texture Extractor Service.

Handles background removal from uploaded garment images and
prepares the cleaned image for use as a 3D texture map.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter
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


def _crop_to_opaque_bbox(image: Image.Image) -> Image.Image:
    """Crop an RGBA image to the bounding box of opaque (alpha > 128) pixels."""
    arr = np.array(image)
    if arr.ndim < 3 or arr.shape[2] < 4:
        return image
    opaque = arr[:, :, 3] > 128
    if not np.any(opaque):
        return image
    rows = np.any(opaque, axis=1)
    cols = np.any(opaque, axis=0)
    r_min, r_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    c_min, c_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return image.crop((c_min, r_min, c_max + 1, r_max + 1))


def _harden_alpha(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Binarize alpha channel: above threshold → 255, below → 0.

    Prevents semi-transparent background pixels (rembg artifacts) from
    leaking sky, nets, trees, etc. into the final texture.
    """
    arr = np.array(image)
    if arr.ndim < 3 or arr.shape[2] < 4:
        return image
    alpha = arr[:, :, 3]
    alpha = np.where(alpha >= threshold, 255, 0).astype(np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def _clean_image(image: Image.Image) -> tuple[Image.Image, Image.Image, tuple[int, int, int]]:
    """Shared pre-processing for prepare_texture: harden alpha, erode fringe, crop, flatten."""
    fill_rgb = extract_dominant_color(image, avoid_white=False, use_center_region=False)

    img_copy = _harden_alpha(image.copy().convert("RGBA"), threshold=48)

    try:
        alpha_ch = img_copy.split()[3]
        alpha_ch = alpha_ch.filter(ImageFilter.MinFilter(5))
        img_copy.putalpha(alpha_ch)
    except Exception:
        pass

    img_copy = _crop_to_opaque_bbox(img_copy)
    iw, ih = img_copy.size
    if iw < 1 or ih < 1:
        return Image.new("RGB", (1, 1), fill_rgb), img_copy, fill_rgb

    img_flat = Image.new("RGB", (iw, ih), fill_rgb)
    if img_copy.mode == "RGBA":
        img_flat.paste(img_copy.convert("RGB"), mask=img_copy.split()[3])
    else:
        img_flat = img_copy.convert("RGB")

    return img_flat, img_copy, fill_rgb


def prepare_texture(
    image: Image.Image,
    target_size: tuple[int, int] = (1024, 1024),
    avoid_white_fill: bool = False,
    frontal: bool = False,
    back_image: Image.Image | None = None,
) -> Image.Image:
    """Prepare the garment image as a texture for 3D mapping.

    Parameters
    ----------
    frontal : bool
        When *True* (pants / front-back projection UVs), produces a
        **dual-half texture atlas** – left half for front-facing vertices,
        right half for back-facing vertices.  If *back_image* is supplied
        it fills the right half; otherwise a blurred copy of the front
        is used so front-only details (buttons, fly) don't appear on
        the back of the mesh.

        When *False* (cylindrical UVs, default for shirts), uses a
        two-layer approach: blurred cover-crop base + crisp fit-mode
        overlay centred on the front of the cylindrical mesh.

    back_image : Image.Image, optional
        RGBA image of the garment's back view (already background-removed).
        Used only when *frontal=True*.
    """
    fill_rgb = extract_dominant_color(image, avoid_white=avoid_white_fill, use_center_region=avoid_white_fill)
    if avoid_white_fill and all(c >= 230 for c in fill_rgb):
        fill_rgb = extract_dominant_color(image, avoid_white=True, use_center_region=True)

    if _is_mostly_solid_color(image, threshold_std=3.0):
        return Image.new("RGB", target_size, fill_rgb)

    img_flat, img_copy, _ = _clean_image(image)
    iw, ih = img_flat.size
    if iw < 1 or ih < 1:
        return Image.new("RGB", target_size, fill_rgb)

    back_img_flat: Image.Image | None = None
    if frontal and back_image is not None:
        back_img_flat, _, _ = _clean_image(back_image)

    if frontal:
        return _prepare_frontal_texture(img_flat, img_copy, fill_rgb, target_size, back_img_flat=back_img_flat)
    return _prepare_cylindrical_texture(img_flat, img_copy, fill_rgb, target_size)


def _fill_half(
    img: Image.Image,
    fill_rgb: tuple[int, int, int],
    half_w: int,
    half_h: int,
) -> Image.Image:
    """Scale *img* to cover a (half_w × half_h) region (cover-crop)."""
    iw, ih = img.size
    scale = max(half_w / max(iw, 1), half_h / max(ih, 1))
    new_w = max(1, int(iw * scale))
    new_h = max(1, int(ih * scale))
    img_s = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x0 = max(0, (new_w - half_w) // 2)
    y0 = max(0, (new_h - half_h) // 2)
    return img_s.crop((x0, y0, x0 + half_w, y0 + half_h))


def _prepare_frontal_texture(
    img_flat: Image.Image,
    img_rgba: Image.Image,
    fill_rgb: tuple[int, int, int],
    target_size: tuple[int, int],
    back_img_flat: Image.Image | None = None,
) -> Image.Image:
    """Dual-half texture atlas for pants (front-back projection UVs).

    Layout (matches ``assign_pants_uvs``):

        ┌──────────────┬──────────────┐
        │  LEFT HALF   │  RIGHT HALF  │
        │  Front view  │  Back view   │
        │  U = 0 – 0.5 │  U = 0.5 – 1 │
        └──────────────┴──────────────┘

    * **Left half** – the front garment photo, cover-cropped.
    * **Right half** – the back photo (if provided) or a blurred version
      of the front so no front-only details (buttons, fly) leak onto the
      back of the mesh.
    """
    tw, th = target_size
    half_w = tw // 2

    front_half = _fill_half(img_flat, fill_rgb, half_w, th)

    if back_img_flat is not None:
        back_half = _fill_half(back_img_flat, fill_rgb, half_w, th)
    else:
        back_half = front_half.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    canvas = Image.new("RGB", target_size, fill_rgb)
    canvas.paste(front_half, (0, 0))
    canvas.paste(back_half, (half_w, 0))
    return canvas


def _prepare_cylindrical_texture(
    img_flat: Image.Image,
    img_rgba: Image.Image,
    fill_rgb: tuple[int, int, int],
    target_size: tuple[int, int],
) -> Image.Image:
    """Texture for cylindrical UVs (shirts / upper-body).

    Cover-crop + depth-blur: the garment photo is scaled to **fully
    cover** the texture (no fill_rgb gaps), so the fabric wraps
    everywhere — sleeves, collar, back.  A smooth blur gradient fades
    details toward the back (left/right edges of the texture) so
    text / logos stay crisp on the front but don't wrap to the back.
    Edge columns are mirror-blended for a seamless back seam.
    """
    tw, th = target_size
    iw, ih = img_flat.size

    # Cover-crop: scale garment to fully fill the texture
    scale = max(tw / max(iw, 1), th / max(ih, 1))
    cover_w = max(1, int(iw * scale))
    cover_h = max(1, int(ih * scale))
    img_full = img_flat.resize((cover_w, cover_h), Image.Resampling.LANCZOS)
    cx = max(0, (cover_w - tw) // 2)
    cy = max(0, (cover_h - th) // 2)
    crisp = img_full.crop((cx, cy, cx + tw, cy + th))

    # Blurred copy for back / sides
    blur_r = max(1, min(tw, th) // 18)
    blurred = crisp.filter(ImageFilter.GaussianBlur(radius=blur_r))

    arr_crisp = np.array(crisp, dtype=np.float64)
    arr_blur = np.array(blurred, dtype=np.float64)

    # Blend weight: 1.0 at front-center (U≈0.5), 0.0 at back (U≈0/1).
    # Cylindrical UVs map the front to [0.25, 0.75] of the texture width.
    u = np.linspace(0.0, 1.0, tw)
    dist = np.abs(u - 0.5)
    weight = np.clip(1.0 - (dist - 0.18) / 0.14, 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight)  # smooth-step
    w = weight[np.newaxis, :, np.newaxis]

    arr = arr_blur * (1.0 - w) + arr_crisp * w

    # Seamless back: mirror-blend left/right edges (U=0 ≈ U=1)
    seam_w = max(1, tw // 8)
    for i in range(seam_w):
        t = 1.0 - (i / seam_w)
        mirror_col = tw - 1 - i
        left_orig = arr[:, i].copy()
        right_orig = arr[:, mirror_col].copy()
        arr[:, i] = (1 - t) * left_orig + t * right_orig
        arr[:, mirror_col] = (1 - t) * right_orig + t * left_orig

    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")


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
