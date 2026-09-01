#!/usr/bin/env python3
"""Build the LTX-2.5 image-to-video API graph from the official ComfyUI template.

Same treatment as run_ltx25.py, plus: the template's LoadImage lives outside the
subgraph, so it is re-created here and wired into ResizeImageMaskNode.
"""
import argparse
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ui2api import convert  # noqa: E402
from run_ltx25 import TEMPLATES, run  # noqa: E402

# node ids inside the i2v subgraph
N_PROMPT, N_NEG = "376", "373"
N_WIDTH, N_HEIGHT, N_DURATION, N_FPS = "372", "360", "362", "361"
N_SEED_BASE, N_SEED_REFINE = "338", "339"
N_UNET, N_CLIP = "384", "387"
N_TEXTENC_POS, N_CREATE_VIDEO, N_RESIZE = "364", "370", "351"
ENHANCER_NODES = ["380", "382", "383", "393"]
N_LOAD_IMAGE, N_SAVE = "901", "900"


def build(a):
    g = convert(f"{TEMPLATES}/video_ltx2_5_i2v.json", "/tmp/ltx25_i2v_raw_api.json")

    g[N_TEXTENC_POS]["inputs"]["text"] = [N_PROMPT, 0]
    for n in ENHANCER_NODES:
        g.pop(n, None)

    # the first frame comes from a LoadImage that sits outside the subgraph
    g[N_LOAD_IMAGE] = {
        "class_type": "LoadImage",
        "inputs": {"image": a.image, "upload": "image"},
        "_meta": {"title": "LoadImage"},
    }
    # resize_type is a dynamic combo, so the converter cannot map its widgets
    # positionally; restate the template's values explicitly
    g[N_RESIZE]["inputs"].update({
        "input": [N_LOAD_IMAGE, 0],
        "resize_type": "scale longer dimension",
        # dynamic-combo sub-inputs are dot-prefixed by the combo's own id
        "resize_type.longer_size": a.resize_longer,
        "scale_method": "lanczos",
    })

    g[N_PROMPT]["inputs"]["value"] = a.prompt
    g[N_NEG]["inputs"]["text"] = a.negative
    g[N_WIDTH]["inputs"]["value"] = a.width
    g[N_HEIGHT]["inputs"]["value"] = a.height
    g[N_DURATION]["inputs"]["value"] = a.duration
    g[N_FPS]["inputs"]["value"] = a.fps
    g[N_SEED_BASE]["inputs"]["noise_seed"] = a.seed
    g[N_SEED_REFINE]["inputs"]["noise_seed"] = a.seed + 1
    g[N_UNET]["inputs"]["unet_name"] = a.unet
    g[N_CLIP]["inputs"]["clip_name"] = a.clip

    g[N_SAVE] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": [N_CREATE_VIDEO, 0],
            "filename_prefix": a.prefix,
            "format": "auto",
            "codec": "auto",
        },
        "_meta": {"title": "SaveVideo"},
    }
    return g


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="filename inside ComfyUI's input directory")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="pc game, console game, video game, cartoon, childish, ugly")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--duration", type=int, default=5)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unet", default="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors")
    p.add_argument("--clip", default="gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors")
    p.add_argument("--resize-longer", type=int, default=1536, dest="resize_longer")
    p.add_argument("--prefix", default="video/LTX_2.5_i2v")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.width % 64 or a.height % 64:
        sys.exit(f"width/height must be divisible by 64: got {a.width}x{a.height}")
    if (a.duration * a.fps + 1) % 8 != 1:
        sys.exit(f"frame count {a.duration * a.fps + 1} must satisfy n % 8 == 1")

    graph = build(a)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ltx25_i2v_api.json")
    json.dump(graph, open(out, "w"), indent=1)
    print(f"graph: {len(graph)} nodes -> {out}")
    if a.dry_run:
        sys.exit(0)
    sys.exit(0 if run(graph, str(uuid.uuid4())) else 1)
