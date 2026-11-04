import json
from pathlib import Path

from agent.dto import ActRequestDTO, ObservationDTO
from agent.planner import (
    Plan,
    PlanStep,
    Requirement,
    check_plan,
    check_requirements,
    drop_redundant_moves,
    resolve_destination,
    simulate,
)

CONTRACTS = Path(__file__).resolve().parents[2] / "docs" / "contracts"


def observation(holding=None, inventory=()):
    data = json.loads((CONTRACTS / "observation.example.json").read_text(encoding="utf-8"))
    data["agent"]["holding"] = holding
    data["agent"]["inventory"] = list(inventory)
    return ObservationDTO.model_validate(data)


def steps(*items):
    return [PlanStep(action=a, label=l, near_label=n) for a, l, n in items]


def test_picking_with_a_full_hand_moves_the_held_object_to_inventory():
    simulation = simulate(
        steps(("pick", "coal_1", None), ("pick", "iron_1", None), ("drop", "coal_1", "furnace_1")),
        observation(),
    )
    assert simulation.errors == []
    assert [r.action for r in simulation.requests] == ["pick", "pick", "drop"]
    assert simulation.holding == "iron_1" and simulation.inventory == []
    assert simulation.placed == {"coal_1": "furnace_1"}


def test_rule_violations_are_reported():
    _, errors = check_plan(steps(("drop", "coal_1", "furnace_1")), observation())
    assert errors and "coal_1 is not carried at this point" in errors[0]

    # 가진 석탄을 철 옆에 두려는데 철부터 주우면 기준 물건을 들게 된다.
    _, errors = check_plan(
        steps(("pick", "iron_1", None), ("drop", "coal_1", "iron_1")),
        observation(inventory=["coal_1"]),
    )
    assert errors and "iron_1 is carried at this point" in errors[0]
    assert "Drop coal_1 next to iron_1 before picking iron_1" in errors[0]

    _, errors = check_plan(steps(("move_to", "cart_1", None)), observation())
    assert errors and "unknown label 'cart_1'" in errors[0]

    _, errors = check_plan(steps(("pick", "furnace_1", None)), observation())
    assert errors and "furnace_1 is fixed and cannot be picked up" in errors[0]


def test_moving_an_object_that_is_already_in_place_is_rejected():
    obs = observation(holding="iron_1")
    coal = next(o for o in obs.objects if o.label == "coal_1")
    coal.position.x, coal.position.z = 1.5, 4.5  # 이미 화로(0, 4) 옆
    _, errors = check_plan(
        steps(("drop", "iron_1", "furnace_1"), ("pick", "coal_1", None), ("drop", "coal_1", "furnace_1")), obs
    )
    assert errors == ["step 3 drop(coal_1, near furnace_1): coal_1 was already next to furnace_1 before this command; leave it where it is"]


def test_requirements_catch_missing_objects():
    # "석탄 1개, 철 2개를 화로 옆에" 인데 철을 1개만 놓는 계획
    plan = steps(
        ("pick", "coal_1", None),
        ("pick", "iron_1", None),
        ("drop", "coal_1", "furnace_1"),
        ("drop", "iron_1", "furnace_1"),
    )
    requirements = [
        Requirement(object_type="coal", count=1, destination="furnace_1"),
        Requirement(object_type="iron", count=2, destination="furnace_1"),
    ]
    obs = observation()
    errors = check_requirements(requirements, simulate(plan, obs), obs, near_radius=3.0)
    assert errors == [
        "requirement '2 iron next to furnace_1' is not met: the plan ends with 1 (iron_1). Add steps for 1 more distinct iron"
    ]

    # 주우라는 요구는 계획 중에 주운 객체와 처음부터 든 객체로 센다.
    carry = [Requirement(object_type="iron", count=1, destination=None)]
    picked = steps(("pick", "iron_1", None),)
    assert check_requirements(carry, simulate(picked, obs), obs, near_radius=3.0) == []
    picked_then_dropped = steps(("pick", "iron_1", None), ("drop", "iron_1", "furnace_1"))
    assert check_requirements(carry, simulate(picked_then_dropped, obs), obs, near_radius=3.0) == []
    assert check_requirements(carry, simulate([], observation(inventory=["iron_2"])), observation(inventory=["iron_2"]), 3.0) == []


def test_destination_text_is_resolved_to_a_label_or_carrying():
    objects = {o.label: o for o in observation().objects}
    assert resolve_destination("next to furnace_1", objects) == "furnace_1"
    assert resolve_destination("carried by agent", objects) is None
    assert resolve_destination("iron_1", objects) == "iron_1"
    assert resolve_destination("ground", objects) == "ground"


def test_put_down_requirement_counts_dropped_objects():
    obs = observation(holding="iron_1")
    requirement = [Requirement(object_type="iron", count=1, destination="ground")]
    assert check_requirements(requirement, simulate([], obs), obs, 3.0) != []
    assert check_requirements(requirement, simulate(steps(("drop", "iron_1", None)), obs), obs, 3.0) == []


def test_moves_right_before_pick_or_drop_are_removed():
    requests, _ = check_plan(
        steps(("move_to", "coal_1", None), ("pick", "coal_1", None), ("move_to", "furnace_1", None),
              ("drop", "coal_1", "furnace_1"), ("move_to", "iron_1", None)),
        observation(),
    )
    assert [(r.action, r.label) for r in drop_redundant_moves(requests)] == [
        ("pick", "coal_1"), ("drop", "coal_1"), ("move_to", "iron_1")
    ]


def test_plan_progress_marks_matching_calls():
    plan = Plan(steps=[ActRequestDTO(action="move_to", label="coal_1"), ActRequestDTO(action="store")])
    plan.mark(ActRequestDTO(action="store"))
    assert plan.next_index() == 0
    assert "-> 1. move_to(coal_1)" in plan.render() and "[x] 2. store()" in plan.render()
