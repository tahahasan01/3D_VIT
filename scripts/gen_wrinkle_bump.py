#!/usr/bin/env python3
"""Generate a tileable grayscale bump map for cloth wrinkles (horizontal + vertical folds).

Output: client/public/textures/wrinkles/wrinkle_bump.png
White = raised, black = recessed. Use as bumpMap in Three.js for fold illusion.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "client" / "public" / "textures" / "wrinkles"
SIZE = 512


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "wrinkle_bump.png"

    # Tileable uv in [0,1]; multiple sine waves for horizontal and vertical folds
    u = np.linspace(0, 1, SIZE, dtype=np.float64)
    v = np.linspace(0, 1, SIZE, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    # Horizontal folds (varying with v) + vertical creases (varying with u)
    freq_h = 8
    freq_v = 6
    h_folds = 0.5 + 0.4 * np.sin(2 * math.pi * freq_h * vv)
    v_folds = 0.5 + 0.35 * np.sin(2 * math.pi * freq_v * uu)
    # Blend so both directions visible
    bump = 0.5 * h_folds + 0.5 * v_folds
    # Soften with a low-freq wave for natural look
    soft = 0.5 + 0.1 * np.sin(2 * math.pi * 2 * uu) * np.sin(2 * math.pi * 2 * vv)
    bump = 0.85 * bump + 0.15 * soft
    bump = np.clip(bump, 0.0, 1.0)

    img = (bump * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(out_path)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
