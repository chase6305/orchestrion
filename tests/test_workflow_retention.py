"""An idle scheduler must not retain a completed run's last dispatched inputs."""

import asyncio
import gc
import weakref

import pytest

from orchestrion.workflow import Action, Node, RunStatus, Workflow, WorkflowEngine


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("idempotency_key", [None, "retained"])
def test_scheduler_releases_completed_run_while_another_run_is_active(
    fails, idempotency_key
):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold(context, inputs):
            entered.set()
            await asyncio.wait_for(release.wait(), 2)

        async def finish(context, inputs):
            if fails:
                raise ValueError("device failed")
            return None

        async with WorkflowEngine(
            [Action("hold", hold), Action("finish", finish)]
        ) as engine:
            long_run = engine.submit(Workflow((Node("hold", "hold"),)))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                short_run = engine.submit(
                    Workflow((Node("done", "finish", {"payload": "x" * 100000}),)),
                    idempotency_key=idempotency_key,
                )
                run_ref = weakref.ref(short_run)
                input_ref = weakref.ref(short_run._by_id["done"])
                expected = RunStatus.FAILED if fails else RunStatus.SUCCEEDED
                assert (await short_run.wait(1)).status is expected
                del short_run
                # Let worker completion callbacks drain before checking retention.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                gc.collect()
                assert input_ref() is None
                assert (run_ref() is None) == (idempotency_key is None)
                assert not long_run.done
            finally:
                release.set()
                await long_run.wait(1)

    asyncio.run(scenario())


def test_background_child_context_does_not_retain_a_completed_engine():
    async def scenario():
        release = asyncio.Event()
        background = []

        async def observer():
            await release.wait()

        async def work(context, inputs):
            background.append(asyncio.create_task(observer()))

        try:
            engine = WorkflowEngine([Action("work", work)])
            reference = weakref.ref(engine)
            async with engine:
                run = engine.submit(Workflow((Node("n", "work"),)))
                await run.wait(1)
            del run, engine
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            gc.collect()
            assert reference() is None
            assert not background[0].done()
        finally:
            release.set()
            await asyncio.gather(*background)

    asyncio.run(scenario())
