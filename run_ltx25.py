#!/usr/bin/env python3
"""Build the LTX-2.5 t2v API graph from the official ComfyUI template and run it.

Differences from the shipped template:
  * the Prompt Enhancer branch (TextGenerateLTX2Prompt + its gemma4-e2b CLIP) is
    dropped, so the extra text-encoder checkpoint is not needed;
  * a SaveVideo node is added (in the template it lives outside the subgraph).
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ui2api import convert  # noqa: E402

from comfy_paths import comfy_server

SERVER = comfy_server()
from comfy_paths import templates_dir

TEMPLATES = templates_dir()

# node ids inside the template's subgraph
N_PROMPT, N_NEG = "376", "373"
N_WIDTH, N_HEIGHT, N_DURATION, N_FPS = "372", "360", "362", "361"
N_SEED_BASE, N_SEED_REFINE = "338", "339"
N_UNET, N_CLIP, N_VIDEO_VAE, N_AUDIO_VAE = "384", "387", "385", "386"
N_TEXTENC_POS, N_CREATE_VIDEO = "364", "370"
ENHANCER_NODES = ["380", "382", "383", "393"]


def post(path, payload):
    req = urllib.request.Request(
        f"{SERVER}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=120))


def get(path):
    with urllib.request.urlopen(f"{SERVER}{path}", timeout=120) as r:
        return json.load(r)


def build(a):
    src = f"{TEMPLATES}/video_ltx2_5_t2v.json"
    tmp = "/tmp/ltx25_raw_api.json"
    g = convert(src, tmp)

    # drop the prompt-enhancer branch, feed the raw prompt straight to the encoder
    g[N_TEXTENC_POS]["inputs"]["text"] = [N_PROMPT, 0]
    for n in ENHANCER_NODES:
        g.pop(n, None)

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

    g["900"] = {
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


def run(graph, client_id):
    t0 = time.time()
    res = post("/prompt", {"prompt": graph, "client_id": client_id})
    pid = res["prompt_id"]
    print(f"queued prompt_id={pid}")
    last = None
    while True:
        hist = get(f"/history/{pid}")
        if pid in hist:
            h = hist[pid]
            status = h["status"]
            print(f"\nstatus={status['status_str']} after {time.time()-t0:.1f}s")
            if status["status_str"] != "success":
                for m in status.get("messages", []):
                    print(" ", json.dumps(m)[:500])
                return None
            outs = [o for v in h["outputs"].values() for o in v.get("images", []) + v.get("videos", [])]
            for o in outs:
                print("  output:", o)
            return h
        q = get("/queue")
        running = q["queue_running"]
        cur = running[0][1] if running else None
        if cur != last:
            last = cur
            print(f"  running={bool(running)} pending={len(q['queue_pending'])} t={time.time()-t0:.0f}s")
        time.sleep(5)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="A close-up of a golden retriever running through a sunny meadow, "
                                       "cinematic lighting, shallow depth of field, the dog barks happily")
    p.add_argument("--negative", default="pc game, console game, video game, cartoon, childish, ugly")
    # the first sampling stage runs at half resolution and LTX needs each side
    # divisible by 32, so the requested size must be divisible by 64
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--duration", type=int, default=5)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unet", default="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors")
    p.add_argument("--clip", default="gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors")
    p.add_argument("--prefix", default="video/LTX_2.5_t2v")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.width % 64 or a.height % 64:
        sys.exit(f"width/height must be divisible by 64 (half-res stage needs /32): got {a.width}x{a.height}")
    if (a.duration * a.fps + 1) % 8 != 1:
        sys.exit(f"frame count {a.duration * a.fps + 1} must satisfy n % 8 == 1")

    graph = build(a)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ltx25_t2v_api.json")
    json.dump(graph, open(out, "w"), indent=1)
    print(f"graph: {len(graph)} nodes -> {out}")
    if a.dry_run:
        sys.exit(0)
    ok = run(graph, str(uuid.uuid4()))
    sys.exit(0 if ok else 1)
