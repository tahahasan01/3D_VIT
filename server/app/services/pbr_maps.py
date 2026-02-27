"""Lightweight PBR map generator using Pillow + NumPy.

Generates normal and roughness maps from a base colour image so garment
GLBs can carry per-pixel PBR data.  No GPU or ML model required.

Normal map:  Sobel-filter on grayscale → XY gradient → tangent-space normal.
Roughness:   Inverted luminance, biased toward fabric-like roughness (0.7-1.0).
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


def generate_normal_map(
    image: Image.Image,
    strength: float = 1.5,
) -> Image.Image:
    """Sobel-based tangent-space normal map from a base colour image.

    Parameters
    ----------
    image : PIL.Image
        Source texture (any mode; converted to L internally).
    strength : float
        Controls perceived bumpiness.  1.0 is gentle, 2.0+ is pronounced.

    Returns
    -------
    PIL.Image  (RGB, same size as *image*)
    """
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float64) / 255.0

    # Sobel gradients
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    gy[1:-1, :] = arr[2:, :] - arr[:-2, :]

    gx *= strength
    gy *= strength

    normal = np.zeros((*arr.shape, 3), dtype=np.float64)
    normal[:, :, 0] = -gx
    normal[:, :, 1] = -gy
    normal[:, :, 2] = 1.0

    length = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    length = np.maximum(length, 1e-8)
    normal /= length

    # Map [-1,1] → [0,255]
    out = ((normal * 0.5 + 0.5) * 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def generate_roughness_map(
    image: Image.Image,
    base_roughness: float = 0.85,
    variation: float = 0.15,
) -> Image.Image:
    """Luminance-derived roughness map biased for fabric.

    Dark areas → slightly higher roughness (crevices / folds).
    Light areas → slightly lower roughness (raised fibers reflect more).

    Parameters
    ----------
    image : PIL.Image
        Source texture.
    base_roughness : float
        Centre roughness value (0-1).
    variation : float
        Half-range of roughness variation around *base_roughness*.

    Returns
    -------
    PIL.Image  (L, same size as *image*)
    """
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    # Invert: dark pixels → high roughness
    inv = 1.0 - gray
    roughness = base_roughness - variation + inv * 2.0 * variation
    roughness = np.clip(roughness, 0.0, 1.0)
    out = (roughness * 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def generate_ao_map(
    image: Image.Image,
    radius: int = 8,
) -> Image.Image:
    """Pseudo ambient-occlusion from a blurred dark-channel extraction.

    Parameters
    ----------
    image : PIL.Image
        Source texture.
    radius : int
        Gaussian blur radius for soft shadow spread.

    Returns
    -------
    PIL.Image  (L, same size as *image*)
    """
    gray = image.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=radius))
    arr = np.asarray(blurred, dtype=np.float64) / 255.0
    # Bias toward white (subtle darkening only in crevices)
    ao = 0.5 + arr * 0.5
    ao = np.clip(ao, 0.0, 1.0)
    return Image.fromarray((ao * 255).astype(np.uint8), mode="L")
