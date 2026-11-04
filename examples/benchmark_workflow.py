"""Measure workflow overhead for chains, wide graphs, and one busy resource.

Run: python -m examples.benchmark_workflow --counts 1000 5000 --repeats 3
These no-op measurements exclude device, worker-thread, and network latency.
"""

import argparse
import asyncio
import json
import platform
import sqlite3
import statistics
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

from orchestrion.workflow import (
    Action,
    Node,
    Resource,
    ResourceClaim,
    RunStatus,
    SQLiteJournal,
    Workflow,
    WorkflowEngine,
)


async def _noop(context, inputs):
    return None


async def _measure(count, topology, journal=False, journal_mode="auto"):
    workflow = Workflow(
        tuple(
            Node(
                str(i), "noop", after=(str(i - 1),) if topology == "chain" and i else ()
            )
            for i in range(count)
        )
    )
    claims = (ResourceClaim("arm"),) if topology == "contended" else ()
    directory_context = tempfile.TemporaryDirectory() if journal else nullcontext(None)
    with directory_context as directory:
        store_context = (
            SQLiteJournal(
                Path(directory) / "cell.db",
                max_events_per_run=10,
                journal_mode=journal_mode,
            )
            if journal
            else nullcontext(None)
        )
        with store_context as store:
            async with WorkflowEngine(
                [Action("noop", _noop, claims)],
                [Resource("arm")],
                max_nodes_per_run=count,
                max_events_per_run=10,
                journal=store,
            ) as engine:
                started = time.perf_counter()
                run = engine.submit(workflow)
                submitted = time.perf_counter()
                result = await run.wait(120)
                finished = time.perf_counter()
                assert result.status == RunStatus.SUCCEEDED
                return {
                    "submit_seconds": submitted - started,
                    "total_seconds": finished - started,
                }


async def benchmark(counts, repeats, journal=False, journal_mode="auto"):
    rows = []
    for topology in ("chain", "wide", "contended"):
        for count in counts:
            samples = [
                await _measure(count, topology, journal, journal_mode)
                for _ in range(repeats)
            ]
            rows.append(
                {
                    "topology": topology,
                    "nodes": count,
                    **{
                        key: statistics.median(sample[key] for sample in samples)
                        for key in samples[0]
                    },
                }
            )
    return {
        "python": platform.python_version(),
        "repeats": repeats,
        "journal": journal,
        "requested_journal_mode": journal_mode if journal else None,
        "sqlite": sqlite3.sqlite_version,
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--journal",
        action="store_true",
        help="measure synchronous SQLite commits; start with --counts 50",
    )
    parser.add_argument(
        "--journal-mode", choices=("auto", "wal", "delete"), default="auto"
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(count < 1 for count in args.counts):
        parser.error("counts and repeats must be positive")
    print(
        json.dumps(
            asyncio.run(
                benchmark(args.counts, args.repeats, args.journal, args.journal_mode)
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
