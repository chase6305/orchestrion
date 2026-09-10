import asyncio
import time

import pytest

from examples.heterogeneous_workflow import CellPlanner, SimulatedCell, run_demo
from orchestrion.workflow import Node, NodeStatus, Workflow


def test_mobile_manipulation_cell_plans_executes_and_releases_every_resource():
    result = asyncio.run(run_demo())
    assert result["status"] == "succeeded"
    assert len(result["nodes"]) == 8
    assert all(item["used"] == 0 for item in result["resources"]["resources"].values())
    assert not result["resources"]["quarantined"]
    assert result["steps"].index("fixture: open") < result["steps"].index(
        "arm: approach detected pose"
    )
    assert result["steps"].index("arm: stowed") < result["steps"].index(
        "fixture: clamp"
    )
    assert result["steps"].index("fixture: clamp") < result["steps"].index(
        "vision: verify placement"
    )


def test_planner_produces_preview_without_changing_device_state():
    async def scenario():
        cell = SimulatedCell()
        engine = cell.engine()
        plan = await CellPlanner().plan(
            {"station": "assembly"}, {"arm_stowed": True}, engine.describe()
        )
        nodes = {node["node_id"]: node for node in engine.validate(plan)["nodes"]}
        assert set(nodes["transfer"]["after"]) == {"detect", "prepare"}
        assert nodes["transfer"]["claims"]["platform_motion"] == 1
        assert nodes["arrive"]["claims"]["platform_motion"] == 1
        assert nodes["arrive"]["inputs"] == {"station": "assembly"}
        assert nodes["transfer"]["inputs"]["detection"] == {
            "$output": {"node_id": "detect", "path": []}
        }
        assert not cell.log and cell.station == "home"

    asyncio.run(scenario())


def test_repeated_cell_submission_does_not_move_or_clamp_again():
    async def scenario():
        cell = SimulatedCell()
        async with cell.engine() as engine:
            plan = await CellPlanner().plan(
                {"station": "assembly"}, {"arm_stowed": True}, engine.describe()
            )
            run = engine.submit(plan, idempotency_key="assembly-42")
            assert engine.submit(plan, idempotency_key="assembly-42") is run
            result = await run.wait(1)
            steps = list(cell.log)
            replay = engine.submit(plan, idempotency_key="assembly-42")
            assert await replay.wait(1) == result
            assert cell.log == steps
            assert cell.log.count("fixture: clamp") == 1
            assert cell.log.count("arm: approach detected pose") == 1

    asyncio.run(scenario())


def test_invalid_vision_pose_fails_before_arm_or_gripper_motion():
    async def scenario():
        class InvalidVisionCell(SimulatedCell):
            async def detect(self, context, inputs):
                result = await super().detect(context, inputs)
                result["pose"] = [0.4, 0.1]
                return result

        cell = InvalidVisionCell()
        async with cell.engine() as engine:
            plan = await CellPlanner().plan(
                {"station": "assembly"}, {"arm_stowed": True}, engine.describe()
            )
            result = await engine.submit(plan).wait(1)
            assert result.nodes["detect"].status == NodeStatus.FAILED
            assert "ActionOutputError" in result.nodes["detect"].error
            assert result.nodes["transfer"].status == NodeStatus.SKIPPED
            assert cell.arm_stowed and not cell.object_placed
            assert not any(step.startswith(("arm:", "gripper:")) for step in cell.log)

    asyncio.run(scenario())


@pytest.mark.parametrize("offset", [-10, 10])
def test_stale_or_future_vision_time_is_rejected_before_manipulation(offset):
    async def scenario():
        cell = SimulatedCell()
        cell.station = "assembly"
        async with cell.engine() as engine:
            run = engine.submit(
                Workflow(
                    (
                        Node(
                            "transfer",
                            "arm.pick_place",
                            {
                                "station": "assembly",
                                "detection": {
                                    "pose": [1, 2, 3],
                                    "frame_id": "workcell",
                                    "station": "assembly",
                                    "observed_at": time.time() + offset,
                                },
                            },
                        ),
                    )
                )
            )
            result = await run.wait(1)
            assert result.nodes["transfer"].status == NodeStatus.FAILED
            assert "stale or future-dated" in result.nodes["transfer"].error
            assert cell.arm_stowed
            assert not cell.log

    asyncio.run(scenario())


def test_live_precondition_rejects_base_motion_with_extended_arm():
    async def scenario():
        cell = SimulatedCell()
        cell.arm_stowed = False
        async with cell.engine() as engine:
            run = engine.submit(
                Workflow((Node("move", "base.navigate", {"station": "assembly"}),))
            )
            result = await run.wait(1)
            assert result.nodes["move"].status == NodeStatus.FAILED
            assert cell.station == "home"

    asyncio.run(scenario())
