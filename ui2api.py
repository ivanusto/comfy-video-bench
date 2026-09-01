#!/usr/bin/env python3
"""Flatten a ComfyUI subgraph-based template into API (/api/prompt) format.

Handles the single-subgraph case used by the LTX-2.5 templates: the subgraph's
inner nodes become the whole graph, and links coming from the subgraph input
node are dropped (the inner Primitive/Loader nodes already carry the values).
"""
import json
import sys
import urllib.request

from comfy_paths import comfy_server

SERVER = comfy_server()
SKIP_TYPES = {"MarkdownNote", "Note", "PreviewAny", "Reroute"}


def object_info():
    with urllib.request.urlopen(f"{SERVER}/object_info", timeout=120) as r:
        return json.load(r)


PRIMITIVES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def widget_names(spec):
    """Ordered widget input names for a class, mirroring the frontend layout.

    Returns None once a variable-width widget (dynamic combo / autogrow) is hit:
    those consume an unpredictable number of widgets_values slots, so anything
    after them cannot be mapped positionally and must be set explicitly.
    """
    names = []
    inp = spec.get("input", {})
    for section in ("required", "optional"):
        for name, definition in (inp.get(section) or {}).items():
            typ = definition[0]
            opts = definition[1] if len(definition) > 1 and isinstance(definition[1], dict) else {}
            if isinstance(typ, str) and (typ.startswith("COMFY_DYNAMICCOMBO") or typ.startswith("COMFY_AUTOGROW")):
                names.append(None)  # barrier
                continue
            # multi-typed widgets are declared as "FLOAT,INT"
            is_widget = isinstance(typ, list) or set(typ.split(",")) <= PRIMITIVES
            if not is_widget:
                continue
            names.append(name)
            # seed-style inputs render an extra control_after_generate widget
            if opts.get("control_after_generate"):
                names.append(f"__control__{name}")
    return names


def convert(template_path, out_path, overrides=None):
    wf = json.load(open(template_path))
    subs = wf.get("definitions", {}).get("subgraphs", [])
    if len(subs) != 1:
        sys.exit(f"expected exactly 1 subgraph, found {len(subs)}")
    sg = subs[0]
    info = object_info()

    nodes = {str(n["id"]): n for n in sg["nodes"] if n["type"] not in SKIP_TYPES}
    # link id -> (origin node id, origin slot)
    origin = {l["id"]: (str(l["origin_id"]), l["origin_slot"]) for l in sg["links"]}
    inner_ids = {str(n["id"]) for n in sg["nodes"]}

    prompt = {}
    for nid, node in nodes.items():
        cls = node["type"]
        if cls not in info:
            sys.exit(f"node {nid}: unknown class_type {cls!r} (is ComfyUI up to date?)")
        inputs = {}

        # positional widget values
        wnames = widget_names(info[cls])
        wvals = node.get("widgets_values") or []
        if isinstance(wvals, dict):
            inputs.update({k: v for k, v in wvals.items() if not k.startswith("__")})
        else:
            for name, val in zip(wnames, wvals):
                if name is None:
                    break  # variable-width widget: stop positional mapping
                if not name.startswith("__control__"):
                    inputs[name] = val

        # link inputs override widgets
        for slot in node.get("inputs", []):
            lid = slot.get("link")
            if lid is None:
                continue
            src = origin.get(lid)
            if src is None or src[0] not in inner_ids or src[0] not in nodes:
                continue  # from subgraph inputNode or a dropped node: keep widget default
            inputs[slot["name"]] = [src[0], src[1]]

        prompt[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": cls}}

    for path, value in (overrides or {}).items():
        nid, key = path.split(".", 1)
        if nid not in prompt:
            sys.exit(f"override target node {nid} not in graph")
        prompt[nid]["inputs"][key] = value

    json.dump(prompt, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}: {len(prompt)} nodes")
    return prompt


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    ov = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    convert(src, dst, ov)
