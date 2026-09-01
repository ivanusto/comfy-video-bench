# comfy-video-bench

Drivers and measurements for running video models through ComfyUI's HTTP API
rather than its canvas: MiniMax-H3 and LTX-2.5, one process, no browser.

It exists because comparing two ways of generating a clip needs the graph to be
built the same way every time, the timings to come from the server rather than a
stopwatch, and the machine to survive the batch. All three turned out to be
harder than the generating.

## What is here

| File | What it does |
| --- | --- |
| `ui2api.py` | Flattens a ComfyUI UI-format workflow template into the API format `/prompt` wants, resolving widget order from `/object_info` |
| `run_h3.py` | MiniMax-H3 text to video and audio, single pass, optional model-only LoRA |
| `run_h3_upscale.py` | The two-pass shape: render small, upscale the latent, refine at the target canvas |
| `run_ltx25*.py` | LTX-2.5 text to video, image to video, first/last frame |
| `thermal_guard.py` | Interrupts a render before the box heat soaks itself into a power cut |
| `sharpness.py` | Laplacian variance across frames, for spotting an upscale that only interpolated |
| `workflows/` | A frozen API-format graph, so a run does not depend on the template package's node ids |
| `docs/latent-upscale-results.md` | The measurements, and what they say |

## Running

Everything talks to `$COMFY_SERVER`, default `http://127.0.0.1:8188`. The stock
templates are found through the installed `comfyui_workflow_templates_json`
package, so run from ComfyUI's venv or set `COMFY_TEMPLATES`.

```sh
python3 thermal_guard.py &                    # if your machine needs it, see below
python3 run_h3.py --width 864 --height 480 --steps 20 --duration 5 --seed 101
python3 run_h3_upscale.py --width 864 --height 480 --scale 2.0 \
  --base-steps 25 --refine-steps 20 --refine-start 12 --seed 102
python3 run_h3.py --lora minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
  --steps 8 --seed 103
```

## Three traps worth knowing before you measure anything

1. **ComfyUI caches node outputs across executions.** Two runs with the same
   seed reuse the first pass, so a two-pass render measures only its second
   pass. That reported 130 s where the honest number was 198 s, and the error
   flatters the result rather than breaking it. Give every run its own seed and
   check `/history` for `execution_cached` before believing a number.
2. **A `DynamicCombo` input is flattened in the API format.** The parent input
   carries the selected key as a plain string and its sub-inputs are separate
   keys prefixed with the parent's id, `mode.width` rather than a nested dict.
   A nested dict passes validation and then never reaches the node, which fails
   at execution with a missing positional argument.
3. **Check which output slot you are wiring.** `ComfyMathExpression` returns
   `(FLOAT, INT, BOOL)`; a frame count wants slot 1, and slot 0 fails validation
   with a type mismatch some distance from the mistake.

## The thermal guard

Written for a DGX Spark (GB10) that drops power under sustained load with
nothing in the kernel log. The run that killed it reached its plateau by minute
four, GPU 82 to 86 C and thermal zone 91 to 94 C, sat there for seven minutes,
and died at minute 11. So it trips on a soak budget, how long the hottest zone
has been at or above a threshold continuously, rather than on an instantaneous
reading: the plateau is simply where a long render lives, and an instantaneous
82 C threshold interrupted a perfectly healthy run.

**Those numbers describe one machine.** The shape of the rule transfers, the
thresholds do not. Measure your own before trusting them, and if your hardware
does not do this, you do not need the guard at all.

## Measurements

`docs/latent-upscale-results.md` has the numbers: on a GB10 at 864x480, the
turbo LoRA on its own beat the latent two-pass upscale, 125 s against 198 s
against a 333 s direct render, with a far simpler graph. They stack to 98 s, but
the 27 s that buys over turbo alone comes with three more failure modes.

## Licence

MIT.
