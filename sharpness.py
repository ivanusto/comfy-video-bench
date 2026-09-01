#!/usr/bin/env python3
"""Compare high-frequency detail between renders.

A latent upscale that merely interpolates shows up here: the frames stay
plausible but lose the fine structure a native render at that canvas keeps.
Laplacian variance over the luma plane is the cheap proxy, and it is only
meaningful between clips of the same canvas and content.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def frames(path, count=8):
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                        "-vf", f"select='not(mod(n\\,{max(1, 124 // count)}))'",
                        "-vsync", "0", f"{tmp}/f%03d.png"], check=True)
        for f in sorted(Path(tmp).glob("*.png"))[:count]:
            out.append(np.asarray(Image.open(f).convert("L"), dtype=np.float64))
    return out


def lap_var(img):
    p = np.pad(img, 1, mode="edge")
    acc = np.zeros_like(img)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w = LAP[dy + 1, dx + 1]
            if w:
                acc += w * p[1 + dy: 1 + dy + img.shape[0], 1 + dx: 1 + dx + img.shape[1]]
    return float(acc.var())


for path in sys.argv[1:]:
    vals = [lap_var(f) for f in frames(path)]
    print(f"{Path(path).name:34} laplacian variance  mean {np.mean(vals):8.1f}  "
          f"min {min(vals):8.1f}  max {max(vals):8.1f}")
