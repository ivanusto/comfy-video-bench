#!/usr/bin/env python3
"""Build the LTX-2.5 first-last-frame-to-video API graph from the official template.

Unlike t2v/i2v this is a single-stage pipeline: no latent upsampler. The two
guide frames are injected with LTXVAddGuide at frame 0 and frame -1, then
removed again by LTXVCropGuides before decoding.
"""
import argparse
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ui2api import convert  # noqa: E402
from run_ltx25 import TEMPLATES, run  # noqa: E402

N_PROMPT, N_NEG = "252", "217"
N_WIDTH, N_HEIGHT, N_DURATION, N_FPS = "215", "216", "198", "205"
N_SEED = "196"
N_UNET, N_CLIP = "230", "228"
N_TEXTENC_POS, N_CREATE_VIDEO = "222", "218"
N_RESIZE_FIRST, N_RESIZE_LAST = "213", "214"
N_GUIDE_FIRST, N_GUIDE_LAST = "206", "204"
ENHANCER_NODES = ["246", "247", "248", "250"]
N_LOAD_FIRST, N_LOAD_LAST, N_SAVE = "901", "902", "900"


def build(a):
    g = convert(f"{TEMPLATES}/video_ltx2_5_flf2v.json", "/tmp/ltx25_flf2v_raw_api.json")

    g[N_TEXTENC_POS]["inputs"]["text"] = [N_PROMPT, 0]
    for n in ENHANCER_NODES:
        g.pop(n, None)

    for nid, fname in ((N_LOAD_FIRST, a.first), (N_LOAD_LAST, a.last)):
        g[nid] = {"class_type": "LoadImage",
                  "inputs": {"image": fname, "upload": "image"},
                  "_meta": {"title": "LoadImage"}}

    # width/height already arrive as links; the combo key and the widgets that
    # follow it were dropped by the converter, so restate the template's values
    for nid, src in ((N_RESIZE_FIRST, N_LOAD_FIRST), (N_RESIZE_LAST, N_LOAD_LAST)):
        g[nid]["inputs"].update({
            "input": [src, 0],
            "resize_type": "scale dimensions",
            "resize_type.crop": "center",
            "scale_method": "nearest-exact",
        })

    g[N_PROMPT]["inputs"]["value"] = a.prompt
    g[N_NEG]["inputs"]["text"] = a.negative
    g[N_WIDTH]["inputs"]["value"] = a.width
    g[N_HEIGHT]["inputs"]["value"] = a.height
    g[N_DURATION]["inputs"]["value"] = a.duration
    g[N_FPS]["inputs"]["value"] = a.fps
    g[N_SEED]["inputs"]["noise_seed"] = a.seed
    g[N_UNET]["inputs"]["unet_name"] = a.unet
    g[N_CLIP]["inputs"]["clip_name"] = a.clip
    g[N_GUIDE_FIRST]["inputs"]["strength"] = a.guide_strength
    g[N_GUIDE_LAST]["inputs"]["strength"] = a.guide_strength

    g[N_SAVE] = {"class_type": "SaveVideo",
                 "inputs": {"video": [N_CREATE_VIDEO, 0], "filename_prefix": a.prefix,
                            "format": "auto", "codec": "auto"},
                 "_meta": {"title": "SaveVideo"}}
    return g


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--first", required=True, help="first-frame filename in ComfyUI's input dir")
    p.add_argument("--last", required=True, help="last-frame filename in ComfyUI's input dir")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="blurry, out of focus, overexposed, underexposed, "
                                         "low contrast, washed out colors, excessive noise, grainy texture")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--duration", type=int, default=5)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guide-strength", type=float, default=0.7, dest="guide_strength")
    p.add_argument("--unet", default="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors")
    p.add_argument("--clip", default="gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors")
    p.add_argument("--prefix", default="video/LTX_2.5_flf2v")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    # single stage here, so /32 would do, but keep the /64 house rule for
    # consistency with the other two scripts
    if a.width % 32 or a.height % 32:
        sys.exit(f"width/height must be divisible by 32: got {a.width}x{a.height}")
    if (a.duration * a.fps + 1) % 8 != 1:
        sys.exit(f"frame count {a.duration * a.fps + 1} must satisfy n % 8 == 1")

    graph = build(a)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ltx25_flf2v_api.json")
    json.dump(graph, open(out, "w"), indent=1)
    print(f"graph: {len(graph)} nodes -> {out}")
    if a.dry_run:
        sys.exit(0)
    sys.exit(0 if run(graph, str(uuid.uuid4())) else 1)
