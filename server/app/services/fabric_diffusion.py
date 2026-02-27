"""FabricDiffusion wrapper – optional texture rectification.

When enabled, crops a representative fabric patch from the garment image,
runs FabricDiffusion to produce a flat / tileable texture, and returns it.
Falls back gracefully when FabricDiffusion is unavailable.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import (
    FABRIC_DIFFUSION_PYTHON,
    FABRIC_DIFFUSION_ROOT,
    FABRIC_DIFFUSION_TIMEOUT,
    USE_FABRIC_DIFFUSION,
)

logger = logging.getLogger(__name__)


def _crop_center_patch(image: Image.Image, patch_size: int = 256) -> Image.Image:
    """Crop a square patch from the centre of the non-transparent region."""
    if image.mode == "RGBA":
        arr = np.asarray(image)
        alpha = arr[:, :, 3]
        ys, xs = np.where(alpha > 128)
        if len(ys) < 100:
            return image.convert("RGB").resize((patch_size, patch_size))
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
    else:
        y0, x0 = 0, 0
        y1, x1 = image.height - 1, image.width - 1

    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    half = min(y1 - y0, x1 - x0, patch_size * 2) // 4
    half = max(half, patch_size // 2)

    crop = image.crop((cx - half, cy - half, cx + half, cy + half))
    return crop.convert("RGB").resize((patch_size, patch_size))


def rectify_texture(garment_image: Image.Image) -> Image.Image | None:
    """Run FabricDiffusion texture normalization on *garment_image*.

    Returns the rectified (flat, tileable) texture or ``None`` when
    FabricDiffusion is disabled / unavailable.
    """
    if not USE_FABRIC_DIFFUSION:
        return None

    root = FABRIC_DIFFUSION_ROOT
    if not root or not Path(root).is_dir():
        logger.info(
            "FabricDiffusion enabled but FABRIC_DIFFUSION_ROOT (%s) is not a directory",
            root,
        )
        return None

    python_bin = FABRIC_DIFFUSION_PYTHON or "python"
    script = Path(root) / "inference_texture.py"
    if not script.is_file():
        logger.warning("FabricDiffusion inference_texture.py not found at %s", script)
        return None

    patch = _crop_center_patch(garment_image, patch_size=256)

    try:
        with tempfile.TemporaryDirectory(prefix="vit_fabdiff_") as tmp:
            src_dir = Path(tmp) / "src"
            src_dir.mkdir()
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()

            patch.save(str(src_dir / "patch.png"))

            cmd = [
                python_bin,
                str(script),
                "--texture_checkpoint",
                "Yuanhao-Harry-Wang/fabric-diffusion-texture",
                "--src_dir",
                str(src_dir),
                "--save_dir",
                str(out_dir),
                "--n_samples",
                "1",
            ]

            env = os.environ.copy()
            env["FABRIC_DIFFUSION_ROOT"] = root

            result = subprocess.run(
                cmd,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=FABRIC_DIFFUSION_TIMEOUT,
            )

            if result.returncode != 0:
                logger.warning(
                    "FabricDiffusion failed (rc=%d): %s",
                    result.returncode,
                    result.stderr[:500],
                )
                return None

            outputs = list(out_dir.rglob("*.png")) + list(out_dir.rglob("*.jpg"))
            if not outputs:
                logger.warning("FabricDiffusion produced no output images")
                return None

            rectified = Image.open(str(outputs[0])).convert("RGB")
            logger.info("FabricDiffusion rectified texture: %s", rectified.size)
            return rectified

    except subprocess.TimeoutExpired:
        logger.warning("FabricDiffusion timed out after %ds", FABRIC_DIFFUSION_TIMEOUT)
        return None
    except Exception:
        logger.exception("FabricDiffusion error")
        return None
