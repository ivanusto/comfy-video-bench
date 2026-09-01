# H3 latent upscale on GB10: does two-pass pay off?

Measured 2026-09-01 on one DGX Spark (GB10), ComfyUI 0.32.0, MiniMax-H3 with
the pruned int8 DiT (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`), NVFP4
AWQ text encoder. All runs 864x480, 5 s, 124 frames, 24 fps, same prompt.

## The question

The [latent upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
renders small, upscales the 24-channel latent with a trained network, then
refines at the target canvas, skipping the 5B VAE decode and re-encode. Worth
it, or is it cheaper to render at the target canvas directly?

## Results

Server-side execution time from `/history`, every run verified to have nothing
served from ComfyUI's node cache.

| Approach | Steps | Time | vs baseline |
| --- | --- | ---: | ---: |
| Direct render | 20 | 333.3 s | 1.00x |
| Two-pass upscale | 25 base + 8 refine | 197.7 s | 1.69x |
| Two-pass upscale | 25 base + 4 refine | 143.5 s | 2.32x |
| Turbo LoRA, direct | 8 | 125.2 s | 2.66x |
| Turbo LoRA, two-pass | 8 base + 4 refine | 98.3 s | 3.39x |

A refine step at 864x480 costs about 13.7 s, so the base pass plus the latent
upscale plus decode is roughly 90 s of the two-pass total.

## What this says

**The turbo LoRA beats the latent upscale on its own**, and with a far simpler
graph: 125 s against 198 s, one sampler instead of two, no upscaler weights, no
second conditioning node. If only one of the two is worth adopting, it is the
LoRA.

**They stack.** Turbo plus two-pass is the fastest at 98 s, 3.4x the direct
baseline, but the extra 27 s saved over turbo alone buys a graph with three more
failure modes. Worth it for a batch, hard to justify for one clip.

## Quality

Every output decoded cleanly at 864x480 with audio, no ghosting or tearing.
Laplacian variance over 8 frames: direct 20.8, two-pass 26.2, turbo direct 13.2,
turbo two-pass 38.0. **Do not rank the techniques by those numbers.** Each run
used a different seed, so the clips differ in content, and how much grass and
fur fills the frame moves this metric more than the sampling path does. What
they do show is that nothing collapsed: a latent upscale that merely
interpolated would sit far below its base render, and none of these do.

By eye, turbo direct is smooth and slightly soft; turbo two-pass carries more
fine detail and some faint horizontal banding on fast-moving edges, which is
what the upscaler's own issue tracker describes as a temporal-chunk seam.

## Three traps, none of them the node pack's fault

1. **`ComfyMathExpression` returns `(FLOAT, INT, BOOL)`.** The template's frame
   count reads slot 1. Wiring a second `MiniMaxH3ImageToVideo` to slot 0 fails
   validation with a type mismatch on `length`.
2. **A `DynamicCombo` is flattened in the API format.** `mode` carries the
   selected key as a plain string and its sub-inputs are separate keys prefixed
   with the parent id, `mode.width` and `mode.height`. A nested dict **passes
   validation** and then never reaches the node, which fails at execution with
   `execute() missing 1 required positional argument: 'mode'`.
3. **ComfyUI caches node outputs across executions.** With a fixed seed the base
   pass, the AV split and the upscaler are all served from cache, so a two-pass
   run measures only the refine pass. That reported 130 s where the honest
   number was 198 s, and the error flatters the result rather than breaking it.
   Every run here uses its own seed, and `/history` is checked for
   `execution_cached` before a number is believed.

## The turbo LoRA on a quantized checkpoint

`lightx2v/Minimax-h3-Turbo` is a LoRA, not a checkpoint. The ComfyUI bf16 file
is keyed to the base module names, and all 208 of them match the pruned int8
repack this box runs: `lora_A [128, 7168]` and `lora_B [5376, 128]` against a
`[5376, 7168]` I8 weight with its scale. ComfyUI patches it without a single
"lora key not loaded", and the output at 8 steps is coherent. Being int8 and
`comfy_quant` did not stand in the way.

## Thermal limits, which decided the shape of this test

This machine drops power under sustained full load, with nothing in the kernel
log. The run that killed it is sampled minute by minute in `~/h3-calib-hw.log`:
1280x704 at 20 steps reached its plateau by minute four or five, GPU 82 to 86 C
and thermal zone 91 to 94 C, sat there for seven minutes, and the box died at
minute 11.

The trigger is one uninterrupted run, not accumulated work: a batch of shorter
renders with cool-down gates survived the same day at a 75 C peak. So the
measurement canvas here is 864x480, where a run is two to five minutes.

`thermal_guard.py` runs beside every render. It trips on a soak budget, how long
the hottest zone has been at or above 88 C continuously, rather than on an
instantaneous reading, because the plateau is simply where a long render lives.
An earlier instantaneous 82 C threshold interrupted a perfectly healthy run. The
fatal run had 417 s of soak; the budget is 240 s. It has since interrupted one
run at the zone ceiling, GPU 85 C and zone 96 C, and the machine stayed up.

## Reproducing

```sh
python3 thermal_guard.py &                       # always
python3 run_h3.py --width 864 --height 480 --steps 20 --duration 5 --seed 101
python3 run_h3_upscale.py --width 864 --height 480 --scale 2.0 --duration 5 \
  --base-steps 25 --refine-steps 20 --refine-start 12 --seed 102
python3 run_h3_upscale.py --lora minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
  --base-steps 8 --refine-steps 8 --refine-start 4 --seed 103
```

Node pack pinned at `d7c01b9`. Upscaler weights in
`~/ComfyUI/models/latent_upscale_models/`, not on the NAS: the pack does not
read `extra_model_paths.yaml` (its issue #35).
