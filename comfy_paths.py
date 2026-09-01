#!/usr/bin/env python3
"""Locate the ComfyUI server and its stock workflow templates."""
import os


def comfy_server():
    return os.environ.get("COMFY_SERVER", "http://127.0.0.1:8188")


def templates_dir():
    """Where ComfyUI's stock workflow templates live.

    Set COMFY_TEMPLATES to override. Otherwise this asks the installed package,
    which is where ComfyUI itself gets them, so it follows a venv upgrade
    instead of pointing at wherever they were the day this was written.
    """
    override = os.environ.get("COMFY_TEMPLATES")
    if override:
        return override
    try:
        import comfyui_workflow_templates_json as pkg
    except ImportError:
        raise SystemExit(
            "comfyui_workflow_templates_json is not importable here. Run this "
            "from ComfyUI's venv, or set COMFY_TEMPLATES to its templates "
            "directory.")
    return os.path.join(os.path.dirname(pkg.__file__), "templates")
