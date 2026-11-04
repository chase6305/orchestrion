"""Measure enumeration construction, optionally against a local Git revision."""

import argparse
import json
import statistics
import subprocess
import time
import types
from pathlib import Path

from orchestrion.schema import FieldSpec


def measure(field_type, count: int, value_type: str, repeats: int) -> float:
    """Return median construction seconds, excluding candidate generation."""
    choices = (
        list(range(count))
        if value_type == "integer"
        else [{"value": [index]} for index in range(count)]
    )
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        field_type(value_type, choices=choices)
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", nargs="+", type=int, default=[1000, 3000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference", help="Trusted local Git revision to compare")
    args = parser.parse_args()
    if args.repeats < 1 or any(count < 1 for count in args.counts):
        parser.error("counts and repeats must be positive")

    implementations = {"working_tree": FieldSpec}
    if args.reference:
        source = subprocess.run(
            ["git", "show", args.reference + ":orchestrion/schema.py"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        module = types.ModuleType("orchestrion_schema_reference")
        exec(compile(source, "<reference schema>", "exec"), module.__dict__)
        implementations[args.reference] = module.FieldSpec

    for value_type in ("integer", "object"):
        for count in args.counts:
            print(json.dumps({
                "type": value_type,
                "choices": count,
                "median_seconds": {
                    name: measure(field_type, count, value_type, args.repeats)
                    for name, field_type in implementations.items()
                },
            }))


if __name__ == "__main__":
    main()
