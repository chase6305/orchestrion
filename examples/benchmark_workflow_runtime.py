"""Measure input compilation, eager cancellation and scoped checkpoint capture.

Run with Python 3.12 for eager cancellation measurements:
python -m examples.benchmark_workflow_runtime --counts 512 2048 --repeats 3
No device SDK, worker pool or network service is started by these probes.
"""

import argparse
import asyncio
import json
import platform
import statistics
import time

from orchestrion.workflow import (
    Action,
    Node,
    OutputRef,
    Resource,
    ResourceClaim,
    Workflow,
    WorkflowEngine,
    workflow_action,
)

_TIMING_FIELDS = (
    "submit_seconds",
    "run_seconds",
    "total_seconds",
    "drain_seconds",
    "total_including_drain_seconds",
)


async def _noop(context, inputs):
    return None


def _compilation(count, payload_items, references, repeats):
    payload = {"values": list(range(payload_items))}
    nodes = [Node("source", "noop")] if references else []
    for index in range(count):
        inputs = dict(payload)
        if references:
            inputs["source"] = OutputRef("source")
        nodes.append(Node(str(index), "noop", inputs))
    workflow = Workflow(tuple(nodes))
    engine = WorkflowEngine([Action("noop", _noop)], max_nodes_per_run=len(nodes))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        engine.validate(workflow)
        samples.append(time.perf_counter() - started)
    return {
        "nodes": count,
        "payload_items": payload_items,
        "output_references": references,
        "median_seconds": statistics.median(samples),
    }


async def _cancel_burst(count):
    loop = asyncio.get_running_loop()
    factory = getattr(asyncio, "eager_task_factory", None)
    if factory is None:
        return {"nodes": count, "supported": False}
    previous_factory = loop.get_task_factory()
    calls, cancellation = [], []
    first_call_at = None

    async def device(context, inputs):
        nonlocal first_call_at
        calls.append(context.node_id)
        if len(calls) == 1:
            first_call_at = time.perf_counter()
            run = engine.get_run_by_key("burst")

            def cancel():
                received_at = time.perf_counter()
                accepted = run.cancel()
                completed_at = time.perf_counter()
                cancellation.append(
                    {
                        "accepted": accepted,
                        "response_seconds": completed_at - first_call_at,
                        "callback_delay_seconds": received_at - first_call_at,
                        "cancellation_seconds": completed_at - received_at,
                    }
                )

            loop.call_soon(cancel)

    engine = WorkflowEngine(
        [Action("device", device, (ResourceClaim("arm"),))],
        [Resource("arm")],
        max_concurrency=1,
        max_nodes_per_run=count,
        max_events_per_run=10,
    )
    try:
        loop.set_task_factory(factory)
        run = engine.submit(
            Workflow(tuple(Node(str(index), "device") for index in range(count))),
            idempotency_key="burst",
        )
        result = await run.wait(30)
        await asyncio.sleep(0)
        return {
            "nodes": count,
            "supported": True,
            "calls_before_cancel": len(calls),
            "status": result.status.value,
            **cancellation[0],
        }
    finally:
        loop.set_task_factory(previous_factory)
        await engine.aclose(5)
        engine.prune_idempotency_keys(0)


async def _checkpoint_capture():
    captures = []

    class SDKRecord(dict):
        def items(self):
            ticket = "capture-" + str(len(captures) + 1)
            captures.append(ticket)
            return {"ticket": ticket}.items()

    async def device(context, inputs):
        context.checkpoint(SDKRecord())

    scoped = workflow_action(
        "scope", Workflow((Node("device", "device"),)), [Action("device", device)]
    )
    async with WorkflowEngine([scoped]) as engine:
        run = engine.submit(Workflow((Node("scope", "scope"),)))
        result = await run.wait(5)
        return {
            "sdk_snapshot_reads": len(captures),
            "recorded_ticket": result.nodes["scope"].checkpoint["scope_nodes"][
                "device"
            ]["ticket"],
        }


async def _with_idle_runs(count, idle_runs, *, eager=False, deadline=False):
    loop = asyncio.get_running_loop()
    factory = getattr(asyncio, "eager_task_factory", None)
    if eager and factory is None:
        return {"nodes": count, "supported": False}
    previous_factory = loop.get_task_factory()
    entered, release = asyncio.Event(), asyncio.Event()
    started = 0

    async def hold(context, inputs):
        nonlocal started
        started += 1
        if started == idle_runs:
            entered.set()
        await release.wait()

    engine = WorkflowEngine(
        [Action("hold", hold), Action("noop", _noop)],
        max_concurrency=idle_runs + 1,
        max_active_runs=idle_runs + 2,
        max_nodes_per_run=count,
        max_events_per_run=10,
    )
    budget = 60 if deadline else None
    workflow = Workflow(
        tuple(Node(str(index), "noop") for index in range(count)), timeout=budget
    )
    try:
        if eager:
            loop.set_task_factory(factory)
        for _ in range(idle_runs):
            engine.submit(Workflow((Node("hold", "hold"),), timeout=budget))
        if idle_runs:
            await asyncio.wait_for(entered.wait(), 5)
        before = time.perf_counter()
        run = engine.submit(workflow)
        submitted = time.perf_counter()
        await run.wait(30)
        completed = time.perf_counter()
        drained = None
        if not idle_runs:
            # Eager completion can publish results before worker done callbacks
            # run. Include those callbacks when comparing completed throughput.
            await engine.aclose(5)
            drained = time.perf_counter()
        result = {
            "nodes": count,
            "supported": True,
            "eager": eager,
            "idle_runs": idle_runs,
            "deadline_seconds": budget,
            "submit_seconds": submitted - before,
            "run_seconds": completed - submitted,
            "total_seconds": completed - before,
        }
        if drained is not None:
            result["drain_seconds"] = drained - completed
            result["total_including_drain_seconds"] = drained - before
        return result
    finally:
        if eager:
            loop.set_task_factory(previous_factory)
        release.set()
        await engine.aclose(5)


async def benchmark(counts, payload_items, repeats, idle_runs):
    compilation, cancellation, active_runs = [], [], []
    eager_completion = []
    for count in counts:
        for references in (False, True):
            compilation.append(_compilation(count, payload_items, references, repeats))
            await asyncio.sleep(0)
        samples = [await _cancel_burst(count) for _ in range(repeats)]
        row = dict(samples[0])
        if row["supported"]:
            for metric in (
                "response_seconds",
                "callback_delay_seconds",
                "cancellation_seconds",
            ):
                row[metric] = statistics.median(sample[metric] for sample in samples)
            row["calls_before_cancel"] = statistics.median(
                sample["calls_before_cancel"] for sample in samples
            )
        cancellation.append(row)
        for deadline in (False, True):
            for held in (0, idle_runs):
                samples = [
                    await _with_idle_runs(count, held, deadline=deadline)
                    for _ in range(repeats)
                ]
                row = dict(samples[0])
                for metric in _TIMING_FIELDS:
                    if metric in row:
                        row[metric] = statistics.median(
                            sample[metric] for sample in samples
                        )
                active_runs.append(row)
        samples = [await _with_idle_runs(count, 0, eager=True) for _ in range(repeats)]
        row = dict(samples[0])
        if row["supported"]:
            for metric in _TIMING_FIELDS:
                if metric in row:
                    row[metric] = statistics.median(sample[metric] for sample in samples)
        eager_completion.append(row)
    return {
        "python": platform.python_version(),
        "repeats": repeats,
        "compilation": compilation,
        "cancellation": cancellation,
        "active_runs": active_runs,
        "eager_completion": eager_completion,
        "scoped_checkpoint": await _checkpoint_capture(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--payload-items", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--idle-runs", type=int, default=64)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    if args.repeats < 1 or any(count < 1 for count in args.counts):
        parser.error("counts and repeats must be positive")
    if args.payload_items < 0:
        parser.error("payload-items must be nonnegative")
    if args.idle_runs < 0:
        parser.error("idle-runs must be nonnegative")
    result = asyncio.run(
        benchmark(args.counts, args.payload_items, args.repeats, args.idle_runs)
    )
    result["label"] = args.label
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
