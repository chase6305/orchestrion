"""The hardware-free runtime probe also runs on the supported Python floor."""

import asyncio

from examples.benchmark_workflow_runtime import benchmark


def test_runtime_probe_finishes_without_leaving_background_tasks():
    async def scenario():
        result = await benchmark([8], payload_items=2, repeats=1, idle_runs=2)
        assert {
            (row["idle_runs"], row["deadline_seconds"])
            for row in result["active_runs"]
        } == {(0, None), (2, None), (0, 60), (2, 60)}
        assert all(row["total_seconds"] >= 0 for row in result["active_runs"])
        eager_supported = hasattr(asyncio, "eager_task_factory")
        assert result["cancellation"][0]["supported"] is eager_supported
        assert result["eager_completion"][0]["supported"] is eager_supported
        captured = result["scoped_checkpoint"]
        assert captured["sdk_snapshot_reads"] == 1
        assert captured["recorded_ticket"] == "capture-1"
        assert captured["sdk_progress_reads"] == 1
        assert captured["recorded_progress"] == 1
        await asyncio.sleep(0)
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())
