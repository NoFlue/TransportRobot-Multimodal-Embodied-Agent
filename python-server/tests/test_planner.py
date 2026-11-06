import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from agent.dto import ActRequestDTO, ObservationDTO
from agent.planner import (
    CommandIntent,
    MotionGoal,
    Plan,
    PlanDraft,
    PlanStep,
    Requirement,
    check_plan,
    check_intent,
    check_requirements,
    drop_redundant_moves,
    make_plan,
    resolve_destination,
    simulate,
)
from tests.fakes import ScriptedChatModel

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


@pytest.mark.parametrize("action", ["move_to", "look_at"])
async def test_motion_intent_never_requests_a_cargo_draft(action):
    # S08/S18 회귀: 정상 이동/시선을 운반 요구 생성/수정에 다시 넣지 않는다.
    target = "furnace_1"
    intent = CommandIntent(motion=[MotionGoal(action=action, label=target)], transport=False)
    correct = PlanDraft(requirements=[], steps=[PlanStep(action=action, label=target)])
    wrong = correct.model_copy(update={"requirements": [Requirement(object_type="furnace", count=1, destination=None)]})
    llm = ScriptedChatModel(replies=[AIMessage(content=d.model_dump_json()) for d in (wrong, correct)], prompts=[])
    outcome = await make_plan(llm, "화로 쪽으로 가거나 바라봐", observation(), 3.0, rewritten="motion request", intent=intent)
    assert outcome.errors == [] and outcome.plan.summary() == f"{action}(furnace_1)"
    assert llm.prompts == []
    assert any("Remove the carrying/placement requirements" in error
               for error in check_intent(intent, wrong, simulate(wrong.steps, observation()).requests, observation()))


@pytest.mark.parametrize("action", ["move_to", "look_at"])
async def test_motion_intent_cannot_be_replaced_by_a_cargo_plan(action):
    intent = CommandIntent(motion=[MotionGoal(action=action, label="coal_1")], transport=False)
    bad = PlanDraft(requirements=[], steps=[PlanStep(action="pick", label="coal_1")])
    llm = ScriptedChatModel(replies=[AIMessage(content=bad.model_dump_json())] * 3, prompts=[])
    outcome = await make_plan(llm, "motion request", observation(), 3.0, rewritten="motion request", intent=intent)
    assert outcome.errors == [] and outcome.plan.summary() == f"{action}(coal_1)"
    assert llm.prompts == []


async def test_transport_validation_failure_never_returns_a_cached_plan():
    # 세계 규칙을 통과한 초안도 수량 검사가 실패하면 반환하지 않는다.
    bad = PlanDraft(requirements=[Requirement(object_type="coal", count=2, destination=None)],
                    steps=[PlanStep(action="pick", label="coal_1")])
    invalid = bad.model_copy(update={"steps": [PlanStep(action="pick", label="furnace_1")]})
    llm = ScriptedChatModel(replies=[AIMessage(content=d.model_dump_json()) for d in (bad, invalid, invalid)], prompts=[])
    outcome = await make_plan(llm, "Pick two coal", observation(), 3.0, rewritten="Pick two coal",
                              intent=CommandIntent(motion=[], transport=True))
    assert outcome.plan is None and any("fixed" in error for error in outcome.errors)


async def test_mixed_intent_keeps_explicit_move_before_pick():
    intent = CommandIntent(motion=[MotionGoal(action="move_to", label="coal_1"),
                                   MotionGoal(action="look_at", label="furnace_1")], transport=True)
    draft = PlanDraft(requirements=[Requirement(object_type="coal", count=1, destination=None)],
                      steps=steps(("move_to", "coal_1", None), ("pick", "coal_1", None), ("look_at", "furnace_1", None)))
    llm = ScriptedChatModel(replies=[AIMessage(content=draft.model_dump_json())], prompts=[])
    outcome = await make_plan(llm, "Go to coal_1, pick it up, then face furnace_1", observation(), 3.0,
                              rewritten="mixed request", intent=intent)
    assert outcome.errors == []
    assert [r.action for r in outcome.plan.steps] == ["move_to", "pick", "look_at"]


async def test_intent_reads_original_command_even_if_rewrite_changed_the_subject():
    draft = PlanDraft(requirements=[], steps=[PlanStep(action="move_to", label="furnace_1")])
    intent = CommandIntent(motion=[MotionGoal(action="move_to", label="furnace_1")], transport=False)
    llm = ScriptedChatModel(replies=[AIMessage(content=d.model_dump_json()) for d in (intent, draft)], prompts=[])
    outcome = await make_plan(llm, "화로 furnace_1 앞으로 이동해", observation(), 3.0,
                              rewritten="Move the furnace next to the furnace.")
    assert "Original user command: 화로 furnace_1 앞으로 이동해" in llm.prompts[0][1].content
    assert outcome.intent == intent and outcome.plan.summary() == "move_to(furnace_1)"
    assert len(llm.prompts) == 1


async def test_replan_only_requires_unfinished_motion_goals_in_original_order():
    intent = CommandIntent(motion=[MotionGoal(action="move_to", label="furnace_1"),
                                   MotionGoal(action="look_at", label="coal_1")], transport=False)
    draft = PlanDraft(requirements=[], steps=[PlanStep(action="look_at", label="coal_1")])
    llm = ScriptedChatModel(replies=[AIMessage(content=draft.model_dump_json())], prompts=[])
    outcome = await make_plan(llm, "Go to the furnace, then look at coal_1", observation(), 3.0,
                              rewritten="motion request", intent=intent, completed_motion=1)
    assert outcome.errors == [] and outcome.plan.summary() == "look_at(coal_1)"
    assert outcome.intent == intent  # 재계획이 원래 의도를 덮어쓰지 않는다.


def test_requirement_rejects_an_object_label_used_as_a_type():
    obs = observation()
    errors = check_requirements([Requirement(object_type="coal_2", count=1, destination=None)],
                                simulate(steps(("pick", "coal_2", None)), obs), obs, 3.0)
    assert len(errors) == 1 and "Unknown object type" in errors[0]
