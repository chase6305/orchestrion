"""Backward-compatible entry point for the pick-and-place Viser demo."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from importlib import import_module

from examples.viser.common import run_demo

workflow = import_module("examples.viser.02_pick_and_place").workflow


if __name__ == "__main__":
    run_demo("Viser Modular Robot Task", workflow)
