"""Backward-compatible entry point for the pick-and-place Viser demo."""

from importlib import import_module

from examples.viser.common import run_demo

workflow = import_module("examples.viser.02_pick_and_place").workflow


if __name__ == "__main__":
    run_demo("Viser Modular Robot Task", workflow)
