#!/usr/bin/env python3
"""Two-pass MiniMax-H3 text-to-video: render small, upscale the latent, refine.

Same flattening approach as run_h3.py, then the stock single-pass graph is
rewired into the hires-fix shape the latent upscaler is built for:

    MiniMaxH3ImageToVideo (base canvas) -> pass 1
      -> LTXVSeparateAVLatent
      -> MinimaxH3LatentUpscaler3D on the video latent, audio untouched
      -> LTXVConcatAVLatent
      -> pass 2 on the tail of the schedule (SplitSigmas)
      -> decode

The second pass gets its own MiniMaxH3ImageToVideo at the target canvas. Reusing
the base pass's conditioning is what the upscaler's issue #29 is about: the
conditioning carries the canvas it was built for, and feeding low-res
conditioning to an upscaled latent trips an IndexError inside calc_cond_batch.
"""
import argparse
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ui2api import convert  # noqa: E402
from run_ltx25 import TEMPLATES, run  # noqa: E402

# Stock template ids, same as run_h3.py.
N_UNET, N_SCHED, N_DECODE, N_VAE = "6", "9", "10", "11"
N_CLIP, N_SAMPLER, N_NOISE, N_GUIDER = "13", "14", "15", "16"
N_SELECT, N_ADECODE, N_AVAE, N_CREATE = "17", "23", "24", "91"
N_H3, N_LENGTH, N_DURATION = "104", "107", "111"
# Added here.
N_SPLIT_AV, N_UP, N_JOIN_AV = "901", "905", "906"
N_H3_HI, N_GUIDER2, N_SCHED2, N_SPLIT, N_NOISE2, N_SAMPLER2 = "908", "909", "910", "911", "912", "913"
N_SAVE = "900"


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


def align32(value):
    return max(32, int(round(value / 32)) * 32)


def build(a):
    g = convert(f"{TEMPLATES}/video_minimax_h3_t2v.json", "/tmp/h3_raw_api.json")

    base_w, base_h = align32(a.width / a.scale), align32(a.height / a.scale)
    g[N_H3]["inputs"].update({"prompt": a.prompt, "width": base_w, "height": base_h})
    g[N_DURATION]["inputs"]["value"] = float(a.duration)
    g[N_NOISE]["inputs"]["noise_seed"] = a.seed
    g[N_SCHED]["inputs"]["steps"] = a.base_steps
    g[N_UNET]["inputs"]["unet_name"] = a.unet
    g[N_CLIP]["inputs"]["clip_name"] = a.clip

    g[N_SPLIT_AV] = {"class_type": "LTXVSeparateAVLatent",
                     "inputs": {"av_latent": [N_SAMPLER, 0]}}
    # Its own loader, not ComfyUI's LatentUpscaleModelLoader, and "target
    # dimensions" rather than a multiplier so the canvas is exactly the one the
    # second pass is conditioned for.
    g[N_UP] = {"class_type": "MinimaxH3LatentUpscaler3D",
               # A DynamicCombo is flattened in the API format: the parent input
               # carries the selected key and its sub-inputs are prefixed with
               # the parent's id. ComfyUI reassembles them into the dict the
               # node's execute() expects. A nested dict passes validation and
               # then never reaches the node.
               "inputs": {"latent": [N_SPLIT_AV, 0], "model_name": a.upscaler,
                          "mode": "target dimensions",
                          "mode.width": a.width, "mode.height": a.height,
                          "align": 32,
                          "enable_temporal_chunking": a.chunking,
                          "force_unload": True,
                          "device": "cuda", "precision": a.precision}}
    # The audio latent has no spatial axes to scale, so it is carried across
    # untouched and rejoined.
    g[N_JOIN_AV] = {"class_type": "LTXVConcatAVLatent",
                    "inputs": {"video_latent": [N_UP, 0],
                               "audio_latent": [N_SPLIT_AV, 1]}}

    g[N_H3_HI] = {"class_type": "MiniMaxH3ImageToVideo",
                  "inputs": {"prompt": a.prompt, "width": a.width, "height": a.height,
                             # ComfyMathExpression returns (FLOAT, INT, BOOL);
                             # length wants the INT, which is slot 1.
                             "length": [N_LENGTH, 1],
                             "clip": [N_CLIP, 0], "vae": [N_VAE, 0]}}
    g[N_GUIDER2] = {"class_type": "BasicGuider",
                    "inputs": {"model": [N_UNET, 0], "conditioning": [N_H3_HI, 0]}}
    g[N_SCHED2] = {"class_type": "BasicScheduler",
                   "inputs": {"model": [N_UNET, 0],
                              "scheduler": g[N_SCHED]["inputs"]["scheduler"],
                              "steps": a.refine_steps, "denoise": 1.0}}
    # Only the tail of the schedule: the latent is already denoised, it just
    # carries the base canvas's detail.
    g[N_SPLIT] = {"class_type": "SplitSigmas",
                  "inputs": {"sigmas": [N_SCHED2, 0], "step": a.refine_start}}
    g[N_NOISE2] = {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed + 1}}
    g[N_SAMPLER2] = {"class_type": "SamplerCustomAdvanced",
                     "inputs": {"noise": [N_NOISE2, 0], "guider": [N_GUIDER2, 0],
                                "sampler": [N_SELECT, 0], "sigmas": [N_SPLIT, 1],
                                "latent_image": [N_JOIN_AV, 0]}}

    g[N_DECODE]["inputs"]["samples"] = [N_SAMPLER2, 0]
    g[N_ADECODE]["inputs"]["samples"] = [N_SAMPLER2, 0]

    if a.lora:
        # Both passes sample with the same patched model.
        add_lora(g, N_UNET, [(N_GUIDER, "model"), (N_SCHED, "model"),
                             (N_GUIDER2, "model"), (N_SCHED2, "model")],
                 a.lora, a.lora_strength)

    g[N_SAVE] = {"class_type": "SaveVideo",
                 "inputs": {"video": [N_CREATE, 0], "filename_prefix": a.prefix,
                            "format": "auto", "codec": "auto"}}
    return g, base_w, base_h


def h3_frames(duration, fps=24):
    """The template's own length formula: frames congruent to 5 mod 17."""
    n = max(5, round(duration * fps))
    return n + (5 - (n % 17)) % 17


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="A close-up of a golden retriever running through a sunny meadow, "
                                       "cinematic lighting, shallow depth of field, the dog barks happily")
    p.add_argument("--width", type=int, default=1280, help="target width")
    p.add_argument("--height", type=int, default=704, help="target height")
    p.add_argument("--scale", type=float, default=2.0, help="base canvas divisor")
    p.add_argument("--duration", type=float, default=5)
    p.add_argument("--base-steps", type=int, default=25)
    p.add_argument("--refine-steps", type=int, default=20,
                   help="length of the second pass schedule before splitting")
    p.add_argument("--refine-start", type=int, default=12,
                   help="split point; the second pass runs refine_steps minus this")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--chunking", action="store_true", default=True)
    p.add_argument("--no-chunking", dest="chunking", action="store_false")
    p.add_argument("--upscaler", default="minimax_h3_latent_upscaler_3d_bf16.safetensors")
    p.add_argument("--unet", default="minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    p.add_argument("--clip", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    p.add_argument("--lora", default=None, help="model-only LoRA, e.g. the turbo adapter")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--prefix", default="video/H3_upscale")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.width % 32 or a.height % 32:
        sys.exit(f"width/height must be divisible by 32: got {a.width}x{a.height}")
    if not 0 < a.refine_start < a.refine_steps:
        sys.exit(f"refine-start must be between 1 and refine-steps-1")

    graph, bw, bh = build(a)
    f = h3_frames(a.duration)
    print(f"graph: {len(graph)} nodes | base {bw}x{bh} -> target {a.width}x{a.height} | "
          f"{f} frames ({f/24:.3f}s) | {a.base_steps} base steps + "
          f"{a.refine_steps - a.refine_start} refine steps")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h3_upscale_api.json")
    json.dump(graph, open(out, "w"), indent=1)
    print(f"froze the graph to {out}")
    if a.dry_run:
        sys.exit(0)
    sys.exit(0 if run(graph, str(uuid.uuid4())) else 1)
