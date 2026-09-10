"""Plan and execute a simulated mobile manipulation cell without optional packages.

Run: python -m examples.heterogeneous_workflow
Preview only: python -m examples.heterogeneous_workflow --preview
"""

import argparse
import asyncio
import json
import time
from typing import Any, Dict, Mapping

from orchestrion import FieldSpec, RequestSchema
from orchestrion.workflow import (
    Action,
    ActionContext,
    Node,
    OutputRef,
    Resource,
    ResourceClaim,
    RunResult,
    RunStatus,
    Workflow,
    WorkflowEngine,
)


class CellPlanner:
    """A small rule-based Planner implementation, independent of device drivers."""

    async def plan(
        self,
        goal: Mapping[str, Any],
        world: Mapping[str, Any],
        catalog: Mapping[str, Any],
    ) -> Workflow:
        if not world.get("arm_stowed"):
            raise ValueError("planning requires a confirmed stowed arm")
        station = goal["station"]
        return Workflow(
            (
                Node("announce", "speech.say", {"text": "Starting placement"}),
                Node("arrive", "base.navigate", {"station": station}),
                Node("detect", "vision.detect", after=("arrive",)),
                Node("prepare", "fixture.open", after=("arrive",)),
                Node(
                    "transfer",
                    "arm.pick_place",
                    {
                        "detection": OutputRef("detect"),
                        "station": station,
                    },
                    after=("prepare",),
                    timeout=2,
                ),
                Node("clamp", "fixture.clamp", after=("transfer",)),
                Node("verify", "vision.verify", after=("clamp",)),
                Node(
                    "complete",
                    "speech.say",
                    {
                        "text": "Placement verified",
                        "verification": OutputRef("verify"),
                    },
                ),
            ),
            timeout=5,
        )


class SimulatedCell:
    """Adapters own device facts and check live preconditions under resource claims."""

    def __init__(self) -> None:
        self.station = "home"
        self.arm_stowed = True
        self.fixture_closed = False
        self.object_placed = False
        self.log = []

    async def _step(self, context: ActionContext, message: str) -> None:
        context.report({"step": message})
        self.log.append(message)
        await context.sleep(0.01)

    async def navigate(self, context: ActionContext, inputs: Dict[str, Any]) -> Dict:
        if not self.arm_stowed:
            raise ValueError("base motion requires the arm to be stowed")
        await self._step(context, "base: arrive at " + inputs["station"])
        self.station = inputs["station"]
        return {"station": self.station}

    async def detect(self, context: ActionContext, inputs: Dict[str, Any]) -> Dict:
        await self._step(context, "vision: detect target")
        return {
            "pose": [0.4, 0.1, 0.2],
            "frame_id": "workcell",
            "station": self.station,
            "observed_at": time.time(),
        }

    async def open_fixture(
        self, context: ActionContext, inputs: Dict[str, Any]
    ) -> Dict:
        await self._step(context, "fixture: open")
        self.fixture_closed = False
        return {"open": True}

    async def pick_place(self, context: ActionContext, inputs: Dict[str, Any]) -> Dict:
        observation = inputs["detection"]
        if self.station != inputs["station"] or observation["station"] != self.station:
            raise ValueError("base pose changed since planning; replan required")
        age = time.time() - observation["observed_at"]
        if age < 0 or age > 1:
            raise ValueError(
                "vision observation is stale or future-dated; capture again"
            )
        if observation["frame_id"] != "workcell" or self.fixture_closed:
            raise ValueError("placement preconditions no longer hold")
        self.arm_stowed = False
        try:
            await self._step(context, "arm: approach detected pose")
            await self._step(context, "gripper: grasp")
            await self._step(context, "arm: place in fixture")
            self.object_placed = True
            await self._step(context, "gripper: release")
        finally:
            # Simulated cleanup confirms a stationary, stowed arm. A real adapter
            # must command and confirm safe recovery, or raise ExecutionUncertain.
            await asyncio.sleep(0.01)
            self.arm_stowed = True
            self.log.append("arm: stowed")
        return {"placed": self.object_placed, "arm_stowed": self.arm_stowed}

    async def clamp(self, context: ActionContext, inputs: Dict[str, Any]) -> Dict:
        if not self.object_placed or not self.arm_stowed:
            raise ValueError("cannot clamp before placement and arm withdrawal")
        await self._step(context, "fixture: clamp")
        self.fixture_closed = True
        return {"clamped": True}

    async def verify(self, context: ActionContext, inputs: Dict[str, Any]) -> Dict:
        await self._step(context, "vision: verify placement")
        if not self.fixture_closed or not self.object_placed:
            raise ValueError("placement verification failed")
        return {"passed": True}

    async def say(self, context: ActionContext, inputs: Dict[str, Any]) -> Dict:
        await self._step(context, "speech: " + inputs["text"])
        return {"spoken": True}

    def engine(self) -> WorkflowEngine:
        def claims(*names):
            return tuple(ResourceClaim(name) for name in names)

        detection = RequestSchema(
            {
                "pose": FieldSpec(
                    "array",
                    required=True,
                    items=FieldSpec("number"),
                    min_items=3,
                    max_items=3,
                ),
                "frame_id": FieldSpec("string", required=True, choices=["workcell"]),
                "station": FieldSpec("string", required=True),
                "observed_at": FieldSpec("number", required=True, minimum=0),
            }
        )

        def confirmed(field):
            return RequestSchema(
                {field: FieldSpec("boolean", required=True, choices=[True])}
            )

        return WorkflowEngine(
            [
                Action(
                    "base.navigate",
                    self.navigate,
                    claims("base", "platform_motion"),
                    RequestSchema({"station": FieldSpec("string", required=True)}),
                    output_schema=RequestSchema(
                        {"station": FieldSpec("string", required=True)}
                    ),
                ),
                Action(
                    "vision.detect",
                    self.detect,
                    claims("camera", "gpu"),
                    output_schema=detection,
                ),
                Action(
                    "vision.verify",
                    self.verify,
                    claims("camera", "gpu", "fixture"),
                    output_schema=confirmed("passed"),
                ),
                Action(
                    "fixture.open",
                    self.open_fixture,
                    claims("fixture", "workcell"),
                    output_schema=confirmed("open"),
                ),
                Action(
                    "fixture.clamp",
                    self.clamp,
                    claims("fixture", "workcell"),
                    output_schema=confirmed("clamped"),
                ),
                # The whole approach/grasp/place/withdraw sequence owns one reservation.
                Action(
                    "arm.pick_place",
                    self.pick_place,
                    claims("arm", "gripper", "fixture", "workcell", "platform_motion"),
                    input_schema=RequestSchema(
                        {
                            "detection": FieldSpec(
                                "object", required=True, object_schema=detection
                            ),
                            "station": FieldSpec("string", required=True),
                        }
                    ),
                    output_schema=RequestSchema(
                        {
                            "placed": FieldSpec(
                                "boolean", required=True, choices=[True]
                            ),
                            "arm_stowed": FieldSpec(
                                "boolean", required=True, choices=[True]
                            ),
                        }
                    ),
                ),
                Action(
                    "speech.say",
                    self.say,
                    claims("speaker"),
                    output_schema=confirmed("spoken"),
                ),
            ],
            [
                Resource(name)
                for name in (
                    "base",
                    "arm",
                    "gripper",
                    "fixture",
                    "camera",
                    "speaker",
                    "workcell",
                    "platform_motion",
                )
            ]
            + [Resource("gpu", capacity=2)],
        )


async def run_demo(preview: bool = False) -> Dict[str, Any]:
    cell = SimulatedCell()
    async with cell.engine() as engine:
        workflow = await CellPlanner().plan(
            {"station": "assembly"}, {"arm_stowed": cell.arm_stowed}, engine.describe()
        )
        if preview:
            return engine.validate(workflow)
        run = engine.submit(workflow, idempotency_key="assembly-cycle-demo")
        result: RunResult = await run.wait(6)
        if result.status != RunStatus.SUCCEEDED:
            raise RuntimeError("workflow failed: {}".format(result))
        return {
            "run_id": result.run_id,
            "idempotency_key": run.idempotency_key,
            "status": result.status.value,
            "nodes": {name: node.status.value for name, node in result.nodes.items()},
            "steps": cell.log,
            "resources": engine.resource_snapshot(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview", action="store_true", help="validate without running actions"
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_demo(args.preview)), indent=2))


if __name__ == "__main__":
    main()
