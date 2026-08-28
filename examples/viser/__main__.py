"""Friendly command-line launcher for all Viser demos."""

import argparse
import importlib
from typing import Dict, List, Optional, Tuple

DEMOS: Dict[str, Tuple[str, str]] = {
    "01": ("Joint Sweep", "examples.viser.01_joint_sweep"),
    "02": ("Pick and Place", "examples.viser.02_pick_and_place"),
    "03": ("Segmented Synchronization", "examples.viser.03_segmented_sync"),
    "04": ("Parallel Arm and Gripper", "examples.viser.04_parallel_motion"),
    "05": ("Live Request Timeline", "examples.viser.05_request_timeline"),
    "06": ("Interactive Controls", "examples.viser.06_interactive_controls"),
    "07": ("End-Effector Motion Trail", "examples.viser.07_motion_trail"),
    "08": ("Gripper Laboratory", "examples.viser.08_gripper_lab"),
}


def _print_demos() -> None:
    for demo_id, (title, _) in DEMOS.items():
        print("{}  {}".format(demo_id, title))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m examples.viser",
        description="Launch an Orchestrion Viser demo.",
        add_help=False,
    )
    parser.add_argument("demo", nargs="?", help="demo number, for example 02")
    parser.add_argument("--list", action="store_true", help="list available demos")
    parser.add_argument("-h", "--help", action="store_true", help="show this help")
    args, demo_args = parser.parse_known_args(argv)
    if args.help and args.demo is None:
        parser.print_help()
        return
    if args.list or args.demo is None:
        _print_demos()
        return

    demo_id = args.demo.zfill(2)
    if demo_id not in DEMOS:
        parser.error("unknown demo {!r}; use --list".format(args.demo))
    title, module_name = DEMOS[demo_id]
    module = importlib.import_module(module_name)
    if args.help:
        demo_args.insert(0, "--help")
    module.run_demo("{} · {}".format(demo_id, title), module.workflow, argv=demo_args)


if __name__ == "__main__":
    main()
