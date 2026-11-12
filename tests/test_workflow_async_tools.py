"""Cancellation observers cannot detach transport work that is still draining."""

import asyncio

import pytest

from orchestrion.workflow.async_tools import cancel_and_drain


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_cancel_observer_failure_does_not_finish_before_the_owned_task_exits(
    failure, caplog
):
    async def scenario():
        entered, draining, release, exited = (asyncio.Event() for _ in range(4))

        async def transport():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                draining.set()
                await release.wait()
            finally:
                exited.set()

        def observer():
            raise failure("cancellation observer interrupted")

        task = asyncio.create_task(transport())
        waiter = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            waiter = asyncio.create_task(cancel_and_drain((task,), on_cancel=observer))
            await asyncio.wait_for(draining.wait(), 1)
            waiter.cancel()
            await asyncio.sleep(0)
            assert not waiter.done()
            assert not exited.is_set()
            release.set()
            assert await asyncio.wait_for(waiter, 1)
            assert exited.is_set() and task.done()
            assert "cancellation observer interrupted" in caplog.text
        finally:
            release.set()
            await asyncio.wait_for(
                asyncio.gather(
                    *(item for item in (task, waiter) if item is not None),
                    return_exceptions=True,
                ),
                1,
            )

    asyncio.run(scenario())
