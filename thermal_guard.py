#!/usr/bin/env python3
"""Interrupt a ComfyUI render before this box heat soaks itself into a power cut.

This machine drops power, with nothing in the kernel log, when it is held hot
for long enough. The run that killed it on 2026-09-01 is sampled minute by
minute in ~/h3-calib-hw.log, and the shape is the point:

    minute   gpu avg  gpu max  zone avg  zone max
         3      68.0       79      76.9        86
         4      79.3       84      88.4        91
         5      81.5       85      91.0        93
        ...      ...      ...       ...       ...
        11      81.0       85      91.7        92   <- power cut at 11:06

It reached its plateau by minute four or five and then sat there. It did not
keep climbing into the cut; it soaked at roughly GPU 82 to 86 and zone 91 to 94
for seven minutes and then died. So an instantaneous threshold is the wrong
instrument: that plateau is simply where any long render lives, and tripping on
it kills healthy runs (an 864x480 render was interrupted at 83 C doing nothing
unusual).

What is measured is time spent hot. This trips on a soak budget instead: how
long the hottest thermal zone has been at or above a threshold, continuously.
The fatal run had 417 s of it; the budget here is 240 s, which would have fired
three minutes before the cut. Two instantaneous ceilings sit above the observed
plateau as a backstop for a faster ramp than the one on record.

Losing a render beats losing the box.

    python3 thermal_guard.py &                 # guard until interrupted
    python3 thermal_guard.py --wait-cool 50    # block until it is cool, then exit
"""
import argparse
import glob
import json
import subprocess
import sys
import time
import urllib.request

from comfy_paths import comfy_server

SERVER = comfy_server()


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def zone_temp():
    """Hottest thermal zone in C. On this platform these run ahead of the GPU."""
    hottest = 0
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(path) as fh:
                hottest = max(hottest, int(fh.read().strip()) // 1000)
        except Exception:
            continue
    return hottest or None


def post(path, payload=None):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(SERVER + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read()


def interrupt():
    """Stop the running prompt and drop anything queued behind it."""
    for path, payload in (("/interrupt", {}), ("/queue", {"clear": True})):
        try:
            post(path, payload)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            print(f"guard: {path} failed: {type(exc).__name__}: {exc}", flush=True)


def sample():
    g, z = gpu_temp(), zone_temp()
    return g, z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--soak-zone", type=int, default=88,
                   help="thermal zone temperature that counts as soaking")
    p.add_argument("--soak-seconds", type=int, default=240,
                   help="how long it may soak continuously before being cut")
    p.add_argument("--gpu-ceiling", type=int, default=88,
                   help="instantaneous GPU backstop, above the observed plateau")
    p.add_argument("--zone-ceiling", type=int, default=96,
                   help="instantaneous thermal zone backstop")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--wait-cool", type=int, default=None,
                   help="instead of guarding, block until at or below this, then exit")
    p.add_argument("--wait-timeout", type=int, default=300)
    a = p.parse_args()

    if a.wait_cool is not None:
        waited = 0
        while waited < a.wait_timeout:
            g, z = sample()
            if g is not None and g <= a.wait_cool:
                print(f"cool: GPU {g} C, zone {z} C after {waited:.0f}s", flush=True)
                return 0
            time.sleep(a.interval)
            waited += a.interval
        g, z = sample()
        print(f"still GPU {g} C, zone {z} C after {waited:.0f}s, continuing anyway", flush=True)
        return 0

    print(f"guard: soak budget {a.soak_seconds}s at zone >= {a.soak_zone} C; "
          f"ceilings GPU {a.gpu_ceiling} C, zone {a.zone_ceiling} C", flush=True)
    peak_g = peak_z = 0
    soak = 0.0
    reported = False
    while True:
        g, z = sample()
        if g is not None:
            peak_g = max(peak_g, g)
        if z is not None:
            peak_z = max(peak_z, z)

        # The soak has to be continuous: dropping back below the threshold
        # means the cooling caught up, which is the state that survived.
        if z is not None and z >= a.soak_zone:
            soak += a.interval
            if soak >= a.soak_seconds / 2 and not reported:
                print(f"guard: {soak:.0f}s into the {a.soak_seconds}s soak budget "
                      f"(zone {z} C, GPU {g} C)", flush=True)
                reported = True
        else:
            soak = 0.0
            reported = False

        why = None
        if soak >= a.soak_seconds:
            why = f"soaked {soak:.0f}s at zone >= {a.soak_zone} C"
        elif g is not None and g >= a.gpu_ceiling:
            why = f"GPU ceiling {g} C"
        elif z is not None and z >= a.zone_ceiling:
            why = f"zone ceiling {z} C"
        if why:
            print(f"guard: TRIPPED, {why} (now GPU {g} C zone {z} C, "
                  f"peaks {peak_g}/{peak_z}); interrupting", flush=True)
            interrupt()
            return 1
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
