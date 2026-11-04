from examples.simulated_pick_and_place import run_demo


def test_pick_and_place_commands_observe_their_completed_motion():
    summary = run_demo()

    assert summary["status"] == "succeeded"
    pick, place = summary["steps"]
    assert (pick["station"], pick["gripper"], pick["joints"]) == (
        "PICK", "close", [0.4, 0.2]
    )
    assert (place["station"], place["gripper"], place["joints"]) == (
        "PLACE", "open", [-0.3, 0.5]
    )
    for step in summary["steps"]:
        assert step["timeline"][0] == "queued"
        assert step["timeline"][-2:] == ["running", "succeeded"]
