"""Command-line launcher for hardware-free dual-arm demos."""

import argparse
import importlib
from typing import Dict, List, Optional, Tuple

DEMOS: Dict[str, Tuple[str, str]] = {
    "01": ("Parallel Pick", "examples.dual_arm.01_parallel_pick"),
    "02": ("Payload Handoff", "examples.dual_arm.02_handoff"),
    "03": ("Shared Zone Safety", "examples.dual_arm.03_shared_zone"),
    "04": ("Coordinated Abort", "examples.dual_arm.04_coordinated_abort"),
    "05": ("Remote Inspection", "examples.dual_arm.05_remote_inspection"),
    "06": ("Health Revision Monitor", "examples.dual_arm.06_health_monitor"),
    "07": ("Handoff and Place", "examples.dual_arm.07_handoff_and_place"),
}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m examples.dual_arm",
        description="Run a hardware-free Orchestrion dual-arm demo.",
    )
    parser.add_argument("demo", nargs="?", help="demo number, for example 02")
    parser.add_argument("--list", action="store_true", help="list available demos")
    parser.add_argument(
        "--viser", action="store_true", help="render the workflow with two UR5 arms"
    )
    args, demo_args = parser.parse_known_args(argv)
    if args.list or args.demo is None:
        for demo_id, (title, _) in DEMOS.items():
            print("{}  {}".format(demo_id, title))
        return
    demo_id = args.demo.zfill(2)
    if demo_id not in DEMOS:
        parser.error("unknown demo {!r}; use --list".format(args.demo))
    title, module_name = DEMOS[demo_id]
    module = importlib.import_module(module_name)
    if args.viser:
        from examples.dual_arm.viser import run_demo

        run_demo("{} · {}".format(demo_id, title), module.run, argv=demo_args)
        return
    if demo_args:
        parser.error("unrecognized arguments: {}".format(" ".join(demo_args)))
    module.main()


if __name__ == "__main__":
    main()
