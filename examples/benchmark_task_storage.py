"""Reproducible task-storage microbenchmarks without device or network I/O."""

import argparse
import concurrent.futures
import gc
import json
import platform
import statistics
import subprocess
import time
import types
from pathlib import Path

from orchestrion.tasks import ThreadedPoolFunctionCallTask


class ImmediateExecutor:
    def submit(self, call, *args):
        future = concurrent.futures.Future()
        future.set_result(call(*args))
        return future


class DeferredExecutor:
    def __init__(self):
        self.futures = []

    def submit(self, call, *args):
        future = concurrent.futures.Future()
        self.futures.append(future)
        return future


def measure(task_type, count: int, scenario: str, repeats: int) -> float:
    """Time task storage work; status measures 100 queries after filling storage."""

    class EchoTask(task_type):
        def _call_fn(self, request_id, content):
            return content

    timings = []
    for _ in range(repeats):
        gc.collect()
        deferred = scenario == "retention"
        executor = DeferredExecutor() if deferred else ImmediateExecutor()
        limit = count if scenario == "pruning" else (100 if deferred else -1)
        task = EchoTask(max_result_count=limit)
        task.initialize(executor=executor)
        try:
            started = time.perf_counter()
            for index in range(count):
                task.invoke_async(index * 2)
            if scenario == "pruning":
                started = time.perf_counter()
                for index in range(count):
                    assert task.forget_response(index * 2)
            elif scenario == "status":
                started = time.perf_counter()
                for _ in range(100):
                    assert task.peek_status()["succeeded"] == count
            if deferred:
                for future in executor.futures:
                    future.set_result(None)
            timings.append(time.perf_counter() - started)
        finally:
            task.stop()
    return statistics.median(timings)


def reference_task(reference: str):
    """Load the task implementation from a local Git revision for comparison."""
    repository = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ["git", "show", "{}:orchestrion/tasks/function_call_task.py".format(reference)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType("orchestrion_task_storage_reference")
    exec(
        compile(source, "{}:function_call_task.py".format(reference), "exec"),
        module.__dict__,
    )
    return module.ThreadedPoolFunctionCallTask


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts", nargs="+", type=positive_integer, default=[1000, 5000]
    )
    parser.add_argument("--repeats", type=positive_integer, default=3)
    parser.add_argument(
        "--scenario",
        choices=["all", "sparse", "retention", "pruning", "status"],
        default="all",
    )
    parser.add_argument(
        "--reference", help="compare with a local Git revision, e.g. HEAD"
    )
    args = parser.parse_args()
    implementations = [("working", ThreadedPoolFunctionCallTask)]
    if args.reference:
        implementations.insert(0, (args.reference, reference_task(args.reference)))
    scenarios = (
        ["sparse", "retention", "pruning", "status"]
        if args.scenario == "all" else [args.scenario]
    )
    measurements = []
    for scenario in scenarios:
        for count in args.counts:
            for label, task_type in implementations:
                seconds = measure(task_type, count, scenario, args.repeats)
                measurements.append(
                    {
                        "implementation": label,
                        "scenario": scenario,
                        "requests": count,
                        "median_seconds": round(seconds, 6),
                        **(
                            {"status_queries": 100,
                             "status_queries_per_second": round(100 / seconds)}
                            if scenario == "status"
                            else {"requests_per_second": round(count / seconds)}
                        ),
                    }
                )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "repeats": args.repeats,
                "measurements": measurements,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
