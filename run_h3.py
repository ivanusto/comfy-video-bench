#!/usr/bin/env python3
"""Run the official ComfyUI MiniMax-H3 text-to-video template via the API.

Same flattening approach as run_ltx25.py. H3's graph is much smaller: a single
MiniMaxH3ImageToVideo node produces both the conditioning and the empty latent,
and sampling is a plain 20-step BasicScheduler run (no two-stage upsampling).
"""
import argparse
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ui2api import convert  # noqa: E402
from run_ltx25 import TEMPLATES, run  # noqa: E402

N_H3, N_DURATION, N_SEED = "104", "111", "15"
N_SCHED, N_UNET, N_CLIP, N_CREATE = "9", "6", "13", "91"
N_SAVE = "900"


def build(a):
    g = convert(f"{TEMPLATES}/video_minimax_h3_t2v.json", "/tmp/h3_raw_api.json")

    g[N_H3]["inputs"].update({"prompt": a.prompt, "width": a.width, "height": a.height})
    g[N_DURATION]["inputs"]["value"] = float(a.duration)
    g[N_SEED]["inputs"]["noise_seed"] = a.seed
    g[N_SCHED]["inputs"]["steps"] = a.steps
    g[N_UNET]["inputs"]["unet_name"] = a.unet
    g[N_CLIP]["inputs"]["clip_name"] = a.clip

    if a.lora:
        add_lora(g, N_UNET, [("16", "model"), (N_SCHED, "model")], a.lora, a.lora_strength)

    g[N_SAVE] = {"class_type": "SaveVideo",
                 "inputs": {"video": [N_CREATE, 0], "filename_prefix": a.prefix,
                            "format": "auto", "codec": "auto"},
                 "_meta": {"title": "SaveVideo"}}
    return g


def add_lora(g, model_node, consumers, lora_name, strength):
    """Insert a model-only LoRA between the UNET and everything downstream.

    The turbo LoRA is keyed to the base checkpoint's module names, which the
    pruned int8 repack on this box keeps: all 208 modules match, and ComfyUI
    patches them without a single "lora key not loaded".
    """
    g["950"] = {"class_type": "LoraLoaderModelOnly",
                "inputs": {"model": [model_node, 0], "lora_name": lora_name,
                           "strength_model": strength}}
    for node, field in consumers:
        g[node]["inputs"][field] = ["950", 0]
    return g


def h3_frames(duration, fps=24):
    """The template's own length formula: frames ≡ 5 (mod 17)."""
    n = max(5, round(duration * fps))
    return n + (5 - (n % 17)) % 17


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="A close-up of a golden retriever running through a sunny meadow, "
                                       "cinematic lighting, shallow depth of field, the dog barks happily")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--duration", type=float, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unet", default="minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    p.add_argument("--clip", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    p.add_argument("--lora", default=None, help="model-only LoRA, e.g. the turbo adapter")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--prefix", default="video/H3_t2v")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.width % 32 or a.height % 32:
        sys.exit(f"width/height must be divisible by 32: got {a.width}x{a.height}")

    graph = build(a)
    f = h3_frames(a.duration)
    print(f"graph: {len(graph)} nodes | {a.width}x{a.height} | {f} frames ({f/24:.3f}s) | {a.steps} steps")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h3_t2v_api.json")
    json.dump(graph, open(out, "w"), indent=1)
    if a.dry_run:
        sys.exit(0)
    sys.exit(0 if run(graph, str(uuid.uuid4())) else 1)
