"""명령 생성기 자체 검증: 정답 행동만 실행하는 에이전트는 생성된 단계·조건·이동 허용 수를 모두 통과해야 한다."""

from __future__ import annotations

import random

import pytest

from agent.command_gen import draft
from agent.dto import ObservationDTO
from agent.dto import ActRequestDTO
from agent.experiment import TaskSpec, allowed_moves, completed_steps, count_moved, unmet_goals
from agent.graph import StepRecord

LABELS = [("coal_1", True), ("coal_2", True), ("iron_1", True), ("iron_2", True), ("furnace_1", False)]


def random_world(rng: random.Random) -> dict:
    """WorldRandomizer처럼 객체끼리 4 m 이상 떨어진 배치."""
    points = [(0.0, -22.0)]
    objects = []
    for label, pickable in LABELS:
        while True:
            x, z = rng.uniform(-26, 26), rng.uniform(-26, 26)
            if all((x - a) ** 2 + (z - b) ** 2 >= 16 for a, b in points):
                break
        points.append((x, z))
        objects.append({"label": label, "position": {"x": x, "y": 0.3, "z": z}, "pickable": pickable})
    return {"agent": {"x": 0.0, "z": -22.0}, "holding": None, "inventory": [], "objects": objects}


def observe(world: dict) -> ObservationDTO:
    return ObservationDTO.model_validate({
        "timestamp": 0, "image_png_base64": "",
        "agent": {"position": {"x": world["agent"]["x"], "y": 0.3, "z": world["agent"]["z"]},
                  "holding": world["holding"], "inventory": list(world["inventory"])},
        "objects": world["objects"],
    })


def execute(world: dict, action: dict) -> None:
    """완벽한 에이전트: Unity 행동을 위치·손·인벤토리 규칙대로 흉내 낸다."""
    objects = {o["label"]: o for o in world["objects"]}
    kind, label, near = action["action"], action.get("label"), action.get("near_label")

    def walk_to(target: str, offset: float) -> None:
        p = objects[target]["position"]
        world["agent"] = {"x": p["x"] + offset, "z": p["z"]}

    if kind == "move_to":
        walk_to(label, 1.0)
    elif kind == "pick":
        walk_to(label, 0.5)
        if world["holding"]:
            world["inventory"].append(world["holding"])
        world["holding"] = label
    elif kind == "store":
        world["inventory"].append(world["holding"])
        world["holding"] = None
    elif kind == "drop":
        if world["holding"] == label:
            world["holding"] = None
        else:
            world["inventory"].remove(label)
        if near:
            walk_to(near, 1.2)
            p = objects[near]["position"]
            objects[label]["position"] = {"x": p["x"] - 1.5, "y": 0.3, "z": p["z"]}
        else:
            objects[label]["position"] = {"x": world["agent"]["x"] + 0.8, "y": 0.3, "z": world["agent"]["z"]}


CASES = [("single", c) for c in ("move", "look", "pick", "drop", "store")] + [("double", ""), ("complex", ""), ("abstract", "")]


@pytest.mark.parametrize("generator,category", CASES)
def test_reference_plan_passes_its_own_checks(generator: str, category: str):
    for seed in range(400):
        rng = random.Random(seed)
        world = random_world(rng)
        made = draft(generator, category, rng, observe(world))
        for action in made.setup:
            execute(world, action)
        before = observe(world)
        spec = made.finalize(before)
        spec.pop("category", None)
        task = TaskSpec.model_validate(spec)
        for action in spec["reference"]:
            execute(world, action)
        after = observe(world)
        records = [StepRecord(i, a["action"], {k: a.get(k) for k in ("label", "near_label") if a.get(k)}, 0, 0, 0, True, "")
                   for i, a in enumerate(spec["reference"], 1)]
        allowed = allowed_moves(task, [ActRequestDTO.model_validate(a) for a in made.setup])
        assert completed_steps(task, records) == len(task.steps), (seed, task.command, task.steps)
        assert unmet_goals(task.goals, after, 3.0) == [], (seed, task.command, task.goals)
        assert count_moved(before, after) <= allowed, (seed, task.command)
        if generator == "double":
            assert len(task.steps) == 2
        if generator == "complex":
            assert 3 <= len(task.steps) <= 5


def test_same_seed_gives_same_command():
    world = random_world(random.Random(7))
    first = draft("complex", "", random.Random(99), observe(world)).finalize(observe(world))
    second = draft("complex", "", random.Random(99), observe(world)).finalize(observe(world))
    assert first == second
