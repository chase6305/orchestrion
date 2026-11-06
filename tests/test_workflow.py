"""Workflow dependencies, resource arbitration, failure propagation, and scaling."""

import asyncio
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from orchestrion.workflow import (
    Action,
    BlockingAction,
    ExecutionUncertain,
    FailurePolicy,
    Node,
    NodeStatus,
    OutputRef,
    Resource,
    ResourceClaim,
    RunStatus,
    Workflow,
    WorkflowEngine,
)


async def echo(context, inputs):
    return inputs


async def reached(event):
    await asyncio.wait_for(event.wait(), 1)


def test_dataflow_infers_dependencies_and_snapshots_mutable_values():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        output = {"objects": [{"pose": [1, 2, 3]}]}

        async def vision(context, inputs):
            entered.set()
            await reached(release)
            return output

        async def move(context, inputs):
            inputs["pose"][0] = 99
            return inputs

        engine = WorkflowEngine([Action("vision", vision), Action("arm.move", move)])
        workflow = Workflow(
            (
                Node(
                    "pick",
                    "arm.move",
                    {"pose": OutputRef("detect", ("objects", 0, "pose"))},
                ),
                Node("detect", "vision"),
            )
        )
        assert engine.validate(workflow)["nodes"][1]["after"] == ["detect"]
        async with engine:
            run = engine.submit(workflow)
            workflow.nodes[0].inputs["pose"] = [0]
            await reached(entered)
            release.set()
            result = await run.wait(1)
            output["objects"][0]["pose"][0] = 100
            assert result.status == RunStatus.SUCCEEDED
            assert result.nodes["detect"].output["objects"][0]["pose"] == [1, 2, 3]
            assert result.nodes["pick"].output["pose"] == [99, 2, 3]
            result.nodes["pick"].output.clear()
            assert run.snapshot().nodes["pick"].output["pose"] == [99, 2, 3]

    asyncio.run(scenario())


def test_running_and_completed_snapshots_isolate_outputs_and_checkpoints():
    async def scenario():
        checkpointed, release = asyncio.Event(), asyncio.Event()
        payload = {"ticket": {"parts": [1, 2]}}

        async def device(context, inputs):
            context.checkpoint(payload)
            checkpointed.set()
            await reached(release)
            return payload

        async with WorkflowEngine([Action("device", device, version="v1")]) as engine:
            run = engine.submit(Workflow((Node("move", "device"),)))
            try:
                await reached(checkpointed)
                active = run.snapshot()
                active.nodes["move"].checkpoint["ticket"]["parts"].clear()
                active.nodes.clear()
                assert run.snapshot().nodes["move"].checkpoint == payload
            finally:
                release.set()

            completed = await run.wait(1)
            original = run.snapshot().nodes["move"]
            assert original.action_version == "v1"
            assert original.started_at is not None
            assert original.finished_at is not None
            completed.nodes["move"].output["ticket"]["parts"].append(3)
            completed.nodes["move"].checkpoint["ticket"]["parts"].clear()
            payload["ticket"]["parts"].append(4)
            assert run.snapshot().nodes["move"] == original
            assert original.output == {"ticket": {"parts": [1, 2]}}
            assert original.checkpoint == original.output

    asyncio.run(scenario())


def test_global_capacity_and_unrelated_devices_run_concurrently():
    async def scenario():
        entered, unrelated, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        async def device(context, inputs):
            calls.append(context.run_id)
            entered.set()
            await reached(release)

        async def speech(context, inputs):
            unrelated.set()

        async with WorkflowEngine(
            [
                Action("arm", device, (ResourceClaim("arm"),)),
                Action("speech", speech),
            ],
            [Resource("arm")],
        ) as engine:
            first = engine.submit(Workflow((Node("arm", "arm"),)))
            await reached(entered)
            second = engine.submit(
                Workflow((Node("arm", "arm"), Node("say", "speech")))
            )
            await reached(unrelated)
            assert calls == [first.run_id]
            assert second.snapshot().nodes["arm"].status == NodeStatus.READY
            release.set()
            await first.wait(1)
            await second.wait(1)
            assert calls == [first.run_id, second.run_id]
            assert engine.resource_snapshot()["resources"]["arm"]["used"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("capacity,concurrency", [(3, 2), (2, 3)])
def test_capacity_two_and_global_execution_limit(capacity, concurrency):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        active = 0
        peak = 0
        calls = 0

        async def compute(context, inputs):
            nonlocal active, peak, calls
            calls += 1
            active += 1
            peak = max(peak, active)
            if active == 2:
                entered.set()
            await reached(release)
            active -= 1

        async with WorkflowEngine(
            [Action("compute", compute, (ResourceClaim("gpu"),))],
            [Resource("gpu", capacity)],
            max_concurrency=concurrency,
        ) as engine:
            run = engine.submit(
                Workflow(tuple(Node(str(i), "compute") for i in range(5)))
            )
            await reached(entered)
            assert engine.resource_snapshot()["resources"]["gpu"]["used"] == 2
            release.set()
            assert (await run.wait(1)).status == RunStatus.SUCCEEDED
            assert peak == 2 and calls == 5

    asyncio.run(scenario())


def test_atomic_claims_preserve_fifo_for_overlapping_waiters():
    async def scenario():
        entered, release, unrelated = asyncio.Event(), asyncio.Event(), asyncio.Event()
        order = []

        async def hold(context, inputs):
            entered.set()
            await reached(release)

        async def record(context, inputs):
            order.append(context.node_id)
            if context.node_id == "unrelated":
                unrelated.set()

        async with WorkflowEngine(
            [Action("hold", hold, (ResourceClaim("a"),)), Action("record", record)],
            [Resource("a"), Resource("b"), Resource("c")],
        ) as engine:
            first = engine.submit(Workflow((Node("hold", "hold"),)))
            await reached(entered)
            run = engine.submit(
                Workflow(
                    (
                        Node(
                            "both",
                            "record",
                            claims=(ResourceClaim("a"), ResourceClaim("b")),
                        ),
                        Node("b_only", "record", claims=(ResourceClaim("b"),)),
                        Node("unrelated", "record", claims=(ResourceClaim("c"),)),
                    )
                )
            )
            await reached(unrelated)
            # The blocked node owns neither a nor b. A later b-only node cannot
            # repeatedly jump ahead and starve the two-resource operation.
            assert engine.resource_snapshot()["resources"]["b"]["used"] == 0
            assert order == ["unrelated"]
            release.set()
            await first.wait(1)
            await run.wait(1)
            assert order == ["unrelated", "both", "b_only"]

    asyncio.run(scenario())


def test_priority_across_runs_and_fifo_ties_without_preemption():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        order = []

        async def record(context, inputs):
            order.append(context.node_id)
            if context.node_id == "active":
                entered.set()
                await reached(release)

        async with WorkflowEngine(
            [Action("work", record)], max_concurrency=1
        ) as engine:
            active = engine.submit(Workflow((Node("active", "work"),)))
            await reached(entered)
            first = engine.submit(
                Workflow((Node("low", "work"), Node("high1", "work", priority=10)))
            )
            second = engine.submit(Workflow((Node("high2", "work", priority=10),)))
            release.set()
            await asyncio.gather(active.wait(1), first.wait(1), second.wait(1))
            assert order == ["active", "high1", "high2", "low"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", [ExecutionUncertain("connection lost"), asyncio.CancelledError()]
)
def test_uncertain_operation_quarantines_claims_until_explicit_reconciliation(failure):
    async def scenario():
        calls = []

        async def uncertain(context, inputs):
            raise failure

        async def record(context, inputs):
            calls.append(context.node_id)

        async with WorkflowEngine(
            [
                Action("uncertain", uncertain, (ResourceClaim("base"),)),
                Action("move", record, (ResourceClaim("base"),)),
            ],
            [Resource("base")],
        ) as engine:
            first = engine.submit(Workflow((Node("lost", "uncertain"),)))
            assert (await first.wait(1)).status == RunStatus.NEEDS_ATTENTION
            operation = first.run_id + "/lost"
            assert operation in engine.resource_snapshot()["quarantined"]
            second = engine.submit(Workflow((Node("next", "move"),)))
            with pytest.raises(asyncio.TimeoutError):
                await second.wait(0.01)
            assert calls == []
            with pytest.raises(ValueError):
                engine.confirm_quiescent(operation, evidence="")
            engine.confirm_quiescent(
                operation, evidence="controller query: operation stopped, revision 42"
            )
            assert (await second.wait(1)).status == RunStatus.SUCCEEDED
            assert calls == ["next"]
            assert first.snapshot().nodes["lost"].status == NodeStatus.UNKNOWN
            assert first.events()[-1].kind == "reconciled"
            assert not engine.resource_snapshot()["quarantined"]
            with pytest.raises(KeyError):
                engine.confirm_quiescent(operation, evidence="duplicate")

    asyncio.run(scenario())


def test_fail_fast_cancels_siblings_and_waits_for_cleanup():
    async def scenario():
        entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def move(context, inputs):
            entered.set()
            await context.wait_cancelled()
            stopped.set()
            await reached(release)
            context.raise_if_cancelled()

        async def fail(context, inputs):
            await reached(entered)
            raise RuntimeError("inspection failed")

        async with WorkflowEngine(
            [Action("move", move), Action("fail", fail), Action("echo", echo)]
        ) as engine:
            run = engine.submit(
                Workflow(
                    (
                        Node("arm", "move"),
                        Node("vision", "fail"),
                        Node("grip", "echo", after=("arm",)),
                    )
                )
            )
            await reached(stopped)
            assert not run.done
            assert run.snapshot().nodes["grip"].status == NodeStatus.SKIPPED
            release.set()
            result = await run.wait(1)
            assert result.status == RunStatus.FAILED
            assert result.nodes["arm"].status == NodeStatus.CANCELLED
            assert result.nodes["vision"].status == NodeStatus.FAILED

    asyncio.run(scenario())


def test_continue_independent_only_skips_failed_descendants():
    async def scenario():
        async def fail(context, inputs):
            raise RuntimeError("camera unavailable")

        async with WorkflowEngine(
            [Action("fail", fail), Action("echo", echo)]
        ) as engine:
            run = engine.submit(
                Workflow(
                    (
                        Node("bad", "fail"),
                        Node("descendant", "echo", after=("bad",)),
                        Node("grandchild", "echo", after=("descendant",)),
                        Node("speech", "echo"),
                        Node("done", "echo", after=("speech",)),
                    ),
                    FailurePolicy.CONTINUE_INDEPENDENT,
                )
            )
            result = await run.wait(1)
            assert result.status == RunStatus.FAILED
            assert result.nodes["grandchild"].status == NodeStatus.SKIPPED
            assert result.nodes["done"].status == NodeStatus.SUCCEEDED

    asyncio.run(scenario())


def test_blocking_sdk_does_not_block_loop_or_release_claim_on_cancel():
    async def scenario():
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def sdk(inputs):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(2)
            return {"applied": True}

        with ThreadPoolExecutor(max_workers=1) as executor:
            async with WorkflowEngine(
                [
                    Action(
                        "sdk",
                        BlockingAction(sdk, executor),
                        (ResourceClaim("fixture"),),
                    ),
                    Action("echo", echo),
                ],
                [Resource("fixture")],
            ) as engine:
                try:
                    run = engine.submit(Workflow((Node("clamp", "sdk"),)))
                    await reached(entered)
                    run.cancel()
                    other = engine.submit(Workflow((Node("speech", "echo"),)))
                    assert (await other.wait(1)).status == RunStatus.SUCCEEDED
                    assert (
                        engine.resource_snapshot()["resources"]["fixture"]["used"] == 1
                    )
                    assert not run.done
                finally:
                    release.set()
                result = await run.wait(1)
                assert result.nodes["clamp"].status == NodeStatus.CANCELLED
                # Cancellation does not undo a command that physically completed.
                assert result.nodes["clamp"].output == {"applied": True}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "nodes,match",
    [
        ((Node("x", "absent"),), "unknown action"),
        ((Node("x", "echo"), Node("x", "echo")), "duplicate node"),
        ((Node("x", "echo", after=("absent",)),), "unknown dependency"),
        ((Node("x", "echo", after=("x",)),), "cycle"),
        (
            (
                Node("x", "echo", {"value": OutputRef("y")}),
                Node("y", "echo", after=("x",)),
            ),
            "cycle",
        ),
        ((Node("x", "echo", claims=(ResourceClaim("missing"),)),), "unknown resource"),
        ((Node("x", "echo", claims=(ResourceClaim("gpu", 3),)),), "exceeds capacity"),
        (
            (Node("x", "echo", claims=(ResourceClaim("gpu"), ResourceClaim("gpu"))),),
            "duplicate resource",
        ),
    ],
)
def test_invalid_graph_is_rejected_before_dispatch(nodes, match):
    async def scenario():
        calls = []

        async def record(context, inputs):
            calls.append(inputs)

        async with WorkflowEngine(
            [Action("echo", record)], [Resource("gpu", 2)]
        ) as engine:
            with pytest.raises(ValueError, match=match):
                engine.submit(Workflow((Node("would_run", "echo"), *nodes)))
            assert calls == [] and not engine._runs

    asyncio.run(scenario())


def test_mandatory_action_claim_cannot_be_reduced_by_plan():
    engine = WorkflowEngine(
        [Action("infer", echo, (ResourceClaim("gpu", 2),))], [Resource("gpu", 3)]
    )
    preview = engine.validate(
        Workflow((Node("infer", "infer", claims=(ResourceClaim("gpu"),)),))
    )
    assert preview["nodes"][0]["claims"] == {"gpu": 2}
    preview["nodes"][0]["claims"]["gpu"] = 0
    assert engine.describe()["actions"]["infer"]["claims"] == {"gpu": 2}


def test_engine_cannot_be_reused_across_event_loops():
    engine = WorkflowEngine([Action("echo", echo)])

    async def first():
        await engine.submit(Workflow((Node("x", "echo"),))).wait(1)

    asyncio.run(first())

    async def second():
        with pytest.raises(RuntimeError, match="original event loop"):
            engine.submit(Workflow((Node("x", "echo"),)))

    asyncio.run(second())


def test_deep_workflows_execute_and_propagate_failure_without_recursion():
    async def scenario():
        async def fail(context, inputs):
            raise RuntimeError("first node failed")

        async with WorkflowEngine(
            [Action("echo", echo), Action("fail", fail)]
        ) as engine:
            chain = tuple(
                Node(str(i), "echo", after=(str(i - 1),) if i else ())
                for i in range(3000)
            )
            success = engine.submit(Workflow(chain))
            assert (await success.wait(5)).status == RunStatus.SUCCEEDED
            failed = engine.submit(Workflow((Node("0", "fail"), *chain[1:])))
            result = await failed.wait(5)
            assert result.status == RunStatus.FAILED
            assert result.nodes["2999"].status == NodeStatus.SKIPPED

    asyncio.run(scenario())


@pytest.mark.parametrize("unrelated_resources", [0, 64])
def test_cancelled_waiters_do_not_stall_resource_queue_or_retain_old_runs(
    unrelated_resources,
):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold(context, inputs):
            entered.set()
            await reached(release)

        async with WorkflowEngine(
            [
                Action("hold", hold, (ResourceClaim("arm"),)),
                Action("echo", echo, (ResourceClaim("arm"),)),
            ],
            [Resource("arm")]
            + [Resource("unused-" + str(i)) for i in range(unrelated_resources)],
        ) as engine:
            first = engine.submit(Workflow((Node("hold", "hold"),)))
            await reached(entered)
            blocked = engine.submit(
                Workflow(tuple(Node(str(i), "echo") for i in range(1000)))
            )
            next_run = engine.submit(Workflow((Node("next", "echo"),)))
            blocked.cancel()
            assert (await blocked.wait(1)).status == RunStatus.CANCELLED
            assert len(engine._resource_waiters["arm"]) == 1
            release.set()
            await first.wait(1)
            assert (await next_run.wait(1)).status == RunStatus.SUCCEEDED

    asyncio.run(scenario())


@pytest.mark.parametrize("unrelated_resources", [0, 64])
def test_cancelling_multi_resource_head_wakes_follower_before_owner_finishes(
    unrelated_resources,
):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def hold(context, inputs):
            entered.set()
            await reached(release)

        async def record(context, inputs):
            calls.append(context.node_id)

        async with WorkflowEngine(
            [Action("hold", hold, (ResourceClaim("arm"),)), Action("record", record)],
            [Resource("arm"), Resource("tool")]
            + [Resource("unused-" + str(i)) for i in range(unrelated_resources)],
            max_concurrency=2,
        ) as engine:
            owner = engine.submit(Workflow((Node("owner", "hold"),)))
            try:
                await reached(entered)
                blocked = engine.submit(
                    Workflow(
                        (
                            Node(
                                "both",
                                "record",
                                claims=(ResourceClaim("arm"), ResourceClaim("tool")),
                            ),
                        )
                    )
                )
                follower = engine.submit(
                    Workflow(
                        (Node("follower", "record", claims=(ResourceClaim("tool"),)),)
                    )
                )
                assert blocked.cancel()
                assert (await blocked.wait(1)).status == RunStatus.CANCELLED
                assert (await follower.wait(1)).status == RunStatus.SUCCEEDED
                assert calls == ["follower"]
                assert engine.resource_snapshot()["resources"]["arm"]["used"] == 1
            finally:
                release.set()
            await owner.wait(1)

    asyncio.run(scenario())


def test_run_identity_and_operation_identity_are_read_only():
    async def scenario():
        async def inspect(context, inputs):
            with pytest.raises(AttributeError):
                context.operation_id = "changed"
            with pytest.raises(AttributeError):
                context.run_id = "changed"
            return {"operation": context.operation_id}

        async with WorkflowEngine([Action("inspect", inspect)]) as engine:
            run = engine.submit(Workflow((Node("node", "inspect"),)))
            with pytest.raises(AttributeError):
                run.run_id = "changed"
            result = await run.wait(1)
            assert result.nodes["node"].output["operation"] == run.run_id + "/node"

    asyncio.run(scenario())


@pytest.mark.parametrize("seed", range(10))
def test_randomized_workflows_preserve_capacity_dependencies_and_single_execution(seed):
    async def scenario():
        rng = random.Random(seed)
        capacities = {"arm": 1, "camera": 2, "gpu": 3}
        active = dict.fromkeys(capacities, 0)
        completed = set()
        executed = set()
        manifests = {}

        async def work(context, inputs):
            key = (context.run_id, context.node_id)
            assert key not in executed
            executed.add(key)
            node = manifests[context.run_id][context.node_id]
            assert all(
                (context.run_id, predecessor) in completed for predecessor in node.after
            )
            for claim in node.claims:
                active[claim.resource] += claim.units
                assert active[claim.resource] <= capacities[claim.resource]
            try:
                await context.sleep(inputs["delay"])
                if inputs["fail"]:
                    raise RuntimeError("injected failure")
                completed.add(key)
            finally:
                for claim in node.claims:
                    active[claim.resource] -= claim.units

        async with WorkflowEngine(
            [Action("work", work)],
            [Resource(name, capacity) for name, capacity in capacities.items()],
            max_concurrency=5,
        ) as engine:
            runs = []
            for _ in range(12):
                nodes = []
                for i in range(20):
                    claims = tuple(
                        ResourceClaim(name) for name in capacities if rng.random() < 0.5
                    )
                    after = tuple(str(j) for j in range(i) if rng.random() < 0.06)
                    nodes.append(
                        Node(
                            str(i),
                            "work",
                            {
                                "delay": 0.0001 + rng.random() * 0.0001,
                                "fail": rng.random() < 0.04,
                            },
                            after=after,
                            claims=claims,
                            priority=rng.randrange(3),
                        )
                    )
                workflow = Workflow(tuple(nodes), FailurePolicy.CONTINUE_INDEPENDENT)
                run = engine.submit(workflow)
                manifests[run.run_id] = {node.node_id: node for node in nodes}
                runs.append(run)
            await asyncio.sleep(0.002)
            for run in runs[::4]:
                run.cancel()
            results = await asyncio.gather(*(run.wait(5) for run in runs))
            assert all(
                node.status.terminal
                for result in results
                for node in result.nodes.values()
            )
            assert all(used == 0 for used in active.values())
            assert all(
                item["used"] == 0
                for item in engine.resource_snapshot()["resources"].values()
            )
            for result in results:
                for node in result.nodes.values():
                    assert "AssertionError" not in (node.error or "")
                    if node.status == NodeStatus.FAILED:
                        assert node.error == "RuntimeError: injected failure"
                    if node.status == NodeStatus.SUCCEEDED:
                        assert (result.run_id, node.node_id) in completed
                    if node.status == NodeStatus.SKIPPED:
                        assert (result.run_id, node.node_id) not in executed
            assert not engine._runs

    asyncio.run(scenario())
