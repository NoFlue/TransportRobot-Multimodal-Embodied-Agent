import csv
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.config import load_settings
from agent.metrics import RunRecorder
from agent.service import AgentService
from agent.unity_client import UnityClient
from tests.fakes import FakeUnityWorld, ScriptedChatModel, call

TASK = "석탄 1개와 철 1개를 주워서 화로 주변에 둬"


def make_service(tmp_path: Path, replies: list[AIMessage], judge_replies: list[str], planner_replies: list[str] | None = None):
    settings = load_settings(planner_profile="none")
    world = FakeUnityWorld()
    unity = UnityClient(settings.app.unity, transport=world.transport())
    llm = ScriptedChatModel(replies=replies, prompts=[])
    judge = ScriptedChatModel(replies=[AIMessage(content=text) for text in judge_replies], prompts=[])
    planner = None
    if planner_replies is not None:
        planner = ScriptedChatModel(replies=[AIMessage(content=text) for text in planner_replies], prompts=[])
    recorder = RunRecorder(tmp_path / "runs", save_images=True)
    service = AgentService(settings, unity=unity, llm=llm, judge_llm=judge, planner_llm=planner, recorder=recorder)
    return service, world, llm


def images_in(prompt) -> int:
    return sum(
        1
        for message in prompt
        if isinstance(message, HumanMessage) and not isinstance(message.content, str)
        for part in message.content
        if part.get("type") == "image_url"
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


async def test_carry_task_runs_ora_loop_and_records_metrics(tmp_path):
    replies = [
        call("move_to", label="coal_1"),
        call("pick", label="coal_1"),
        call("store"),
        call("move_to", label="iron_1"),
        call("pick", label="iron_1"),
        call("move_to", label="furnace_1"),
        call("drop", label="coal_1", near_label="furnace_1"),
        call("drop", label="iron_1", near_label="furnace_1"),
        call("finish", summary="석탄과 철을 화로 옆에 두었습니다."),
        AIMessage(content="화로 옆에 coal_1과 iron_1이 있습니다."),
    ]
    verdict = '{"success": true, "rationale": "both are near the furnace"}'
    service, world, llm = make_service(tmp_path, replies, [verdict, verdict])
    await service.start()
    try:
        thread = service.create_thread()
        record = await service.run(thread.thread_id, TASK)
        follow_up = await service.run(thread.thread_id, "지금 화로 옆에 뭐가 있어?")
    finally:
        await service.stop()

    assert record.status == "done" and record.end_reason == "answered"
    assert record.reply == "석탄과 철을 화로 옆에 두었습니다."
    assert record.steps == 8 and record.tool_success_rate == 1.0
    assert record.judge_success is True
    assert record.command_to_first_action_seconds is not None and record.command_to_first_action_seconds >= 0
    assert [c["action"] for c in world.act_calls][:3] == ["move_to", "pick", "store"]
    assert world.holding is None and world.inventory == []

    # 프롬프트에는 최신 관측 이미지 하나만 들어간다.
    assert all(images_in(prompt) == 1 for prompt in llm.prompts)
    # 같은 thread의 다음 명령은 이전 대화를 기억한다.
    assert any(isinstance(m, HumanMessage) and m.content == TASK for m in llm.prompts[-1])
    assert follow_up.reply.startswith("화로 옆에")

    steps = read_csv(service.recorder.run_dir / "steps.csv")
    assert len(steps) == 8
    assert all(float(row["step_to_call_seconds"]) >= 0 for row in steps)
    results = read_csv(service.recorder.run_dir / "results.csv")
    assert [row["judge_success"] for row in results] == ["True", "True"]
    assert len(list((service.recorder.run_dir / "images" / record.command_id).glob("*.png"))) == 8

    roles = [m["role"] for m in thread.messages]
    assert roles.count("action") == 8 and "agent" in roles


async def test_invalid_tool_call_is_reported_back_without_calling_unity(tmp_path):
    replies = [
        call("pick"),  # label 누락
        AIMessage(content="명령을 수행할 수 없습니다."),
    ]
    service, world, llm = make_service(tmp_path, replies, ['{"success": false, "rationale": "nothing moved"}'])
    await service.start()
    try:
        thread = service.create_thread()
        record = await service.run(thread.thread_id, TASK)
    finally:
        await service.stop()

    assert world.act_calls == []
    assert record.steps == 1 and record.tool_success_rate == 0.0
    assert record.judge_success is False
    tool_reply = llm.prompts[1][-2]
    assert tool_reply.content.startswith("failed: Invalid arguments for pick")


async def test_repeated_identical_call_is_not_sent_to_unity(tmp_path):
    replies = [call("look_at", label="furnace_1") for _ in range(3)] + [call("finish", summary="끝")]
    service, world, llm = make_service(tmp_path, replies, ['{"success": false, "rationale": "nothing moved"}'])
    await service.start()
    try:
        thread = service.create_thread()
        record = await service.run(thread.thread_id, "화로를 봐")
    finally:
        await service.stop()

    assert len(world.act_calls) == 2
    assert record.steps == 3 and record.end_reason == "answered"
    assert "call finish" in llm.prompts[3][-2].content
    # 판정에는 명령 시작 시점의 상태가 함께 들어간다.
    judge_prompt = service.judge_llm.prompts[0][1].content[0]["text"]
    assert "Initial world state:" in judge_prompt and "Final world state:" in judge_prompt


async def test_consecutive_failures_trigger_replan_then_stop(tmp_path):
    # 들고 있지 않은 coal_1을 내려놓으려 해서 계속 실패한다. 모델은 같은 호출만 반복한다.
    replies = [call("drop", label="coal_1") for _ in range(6)]
    service, world, llm = make_service(tmp_path, replies, ['{"success": false, "rationale": "nothing moved"}'])
    await service.start()
    try:
        thread = service.create_thread()
        record = await service.run(thread.thread_id, "석탄을 주워")
    finally:
        await service.stop()

    assert record.status == "done" and record.end_reason == "stuck"
    assert record.steps == 6 and len(llm.prompts) == 6  # 7번째 모델 호출 없이 멈춘다
    assert len(world.act_calls) == 2  # 세 번째부터는 반복 방지로 Unity를 부르지 않는다

    def last_observation_text(prompt):
        return prompt[-1].content[0]["text"]

    assert "[recovery]" not in last_observation_text(llm.prompts[2])
    assert "[recovery] Your last 3 actions failed" in last_observation_text(llm.prompts[3])
    assert record.reply.startswith("연속 6번 행동이 실패해서 명령을 중단했습니다.")
    assert [m["role"] for m in thread.messages].count("agent") == 1


def plan_json(*steps) -> str:
    return json.dumps(
        {"requirements": [], "steps": [{"action": a, "label": l, "near_label": n} for a, l, n in steps], "note": ""}
    )


def mentions_json(*objects) -> str:
    return json.dumps({"objects": [{"name": name, "label": label} for name, label in objects]})


async def test_plan_is_executed_without_asking_the_executor_and_uses_history(tmp_path):
    planner_replies = [
        "Put 1 coal next to the furnace.",
        mentions_json(("coal", None), ("furnace", "furnace_1")),
        plan_json(("pick", "coal_1", None), ("drop", "coal_1", "furnace_1")),
        "Put iron_1 next to the furnace.",
        mentions_json(("iron", "iron_1"), ("furnace", "furnace_1")),
        plan_json(("pick", "iron_1", None), ("drop", "iron_1", "furnace_1")),
    ]
    summaries = [AIMessage(content="석탄을 화로 옆에 두었습니다."), AIMessage(content="철도 화로 옆에 두었습니다.")]
    verdict = '{"success": true, "rationale": "ok"}'
    service, world, llm = make_service(tmp_path, summaries, [verdict, verdict], planner_replies=planner_replies)
    await service.start()
    try:
        thread = service.create_thread()
        first = await service.run(thread.thread_id, "석탄 1개를 화로 옆에 둬")
        second = await service.run(thread.thread_id, "철도 거기 둬")
    finally:
        await service.stop()

    assert [(c["action"], c.get("label")) for c in world.act_calls] == [
        ("pick", "coal_1"), ("drop", "coal_1"), ("pick", "iron_1"), ("drop", "iron_1")
    ]
    # 계획을 따르는 동안 실행 모델은 부르지 않고, 끝날 때 요약만 한 번 부른다.
    assert len(llm.prompts) == 2
    assert first.reply == "석탄을 화로 옆에 두었습니다." and first.steps == 2 and first.end_reason == "answered"
    assert second.reply == "철도 화로 옆에 두었습니다."

    rewrite_prompts = [
        p for p in service.planner_llm.prompts if isinstance(p[-1].content, str) and "Latest command:" in p[-1].content
    ]
    assert "Object types in this world: coal, furnace, iron" in rewrite_prompts[0][0].content
    assert rewrite_prompts[0][-1].content == "Latest command: 석탄 1개를 화로 옆에 둬"
    # 두 번째 명령은 이전 대화를 보고 '거기'를 푼다.
    assert "User: 석탄 1개를 화로 옆에 둬" in rewrite_prompts[1][-1].content
    assert "Robot: 석탄을 화로 옆에 두었습니다." in rewrite_prompts[1][-1].content

    plans = [json.loads(line) for line in (service.recorder.run_dir / "plans.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [p["stage"] for p in plans] == ["grounding", "plan", "grounding", "plan"]
    assert all(item["status"] == "known" for item in plans[0]["objects"])
    assert world.view_calls == []  # 모두 아는 객체라 둘러보지 않는다
    plans = [p for p in plans if p["stage"] == "plan"]
    assert [p["rewritten"] for p in plans] == ["Put 1 coal next to the furnace.", "Put iron_1 next to the furnace."]
    assert plans[0]["plan"] == "pick(coal_1) -> drop(coal_1, near furnace_1)"


async def test_failed_plan_step_triggers_an_immediate_replan(tmp_path):
    planner_replies = [
        "Put a coal next to the furnace.",
        mentions_json(("coal", None), ("furnace", "furnace_1")),
        plan_json(("pick", "coal_1", None), ("drop", "coal_1", "furnace_1")),
        plan_json(("pick", "iron_1", None), ("drop", "iron_1", "furnace_1")),
    ]
    service, world, llm = make_service(
        tmp_path, [AIMessage(content="대신 철을 두었습니다.")], ['{"success": false, "rationale": "x"}'], planner_replies
    )
    world.blocked.add(("pick", "coal_1"))
    await service.start()
    try:
        thread = service.create_thread()
        record = await service.run(thread.thread_id, "석탄을 화로 옆에 둬")
    finally:
        await service.stop()

    assert [(c["action"], c.get("label")) for c in world.act_calls] == [
        ("pick", "coal_1"), ("pick", "iron_1"), ("drop", "iron_1")
    ]
    replan_prompt = service.planner_llm.prompts[3][1].content[0]["text"]
    assert replan_prompt.startswith("Command: Put a coal next to the furnace.")
    assert "The previous plan failed" in replan_prompt and "pick(label=coal_1) -> blocked" in replan_prompt
    assert len(llm.prompts) == 1 and record.reply == "대신 철을 두었습니다."
    assert record.steps == 3 and record.tool_success_rate == 2 / 3


async def test_unknown_object_is_found_in_the_view_registered_and_used_by_the_plan(tmp_path):
    planner_replies = [
        "Put 1 coal next to the barrel.",
        # 모델이 통에 화로 label을 붙여도 label은 쓰지 않고 이름으로 판단한다.
        mentions_json(("coal", None), ("barrel", "furnace_1")),
        '{"visible": false}',  # 정면(0°)
        '{"visible": true, "x": 480, "y": 610}',  # 오른쪽 60°
        plan_json(("pick", "coal_1", None), ("drop", "coal_1", "barrel_1")),
    ]
    service, world, llm = make_service(
        tmp_path, [AIMessage(content="통 옆에 석탄을 두었습니다.")], ['{"success": true, "rationale": "ok"}'], planner_replies
    )
    world.hidden["barrel"] = (60.0, [5.0, -18.0])  # 60° 방향에서만 보이는 통
    await service.start()
    try:
        thread = service.create_thread()
        record = await service.run(thread.thread_id, "석탄 1개를 통 옆에 둬")
    finally:
        await service.stop()

    assert world.view_calls == [60.0]  # 정면은 관측 이미지를 그대로 쓴다
    assert world.locate_calls == [{"yaw": 60.0, "x": 0.48, "y": 0.61, "name": "barrel"}]
    assert [(c["action"], c.get("label"), c.get("near_label")) for c in world.act_calls] == [
        ("pick", "coal_1", None), ("drop", "coal_1", "barrel_1")
    ]
    plan_prompt = service.planner_llm.prompts[-1][1].content[0]["text"]
    assert "- barrel_1 type=barrel fixed" in plan_prompt  # 등록된 객체가 관측에 들어간다
    assert "'barrel' = barrel_1" in plan_prompt
    events = [m["text"] for m in thread.messages if m["role"] == "system"]
    assert any("found it by looking around (+60°): barrel_1" in e for e in events)
    assert record.end_reason == "answered" and record.judge_success is True


async def test_object_that_cannot_be_seen_is_reported_to_the_planner(tmp_path):
    planner_replies = [
        "Go to the minecart.",
        mentions_json(("minecart", None)),
        *['{"visible": false}'] * 6,
        json.dumps({"requirements": [], "steps": [], "note": "There is no minecart here."}),
    ]
    service, world, llm = make_service(
        tmp_path, [AIMessage(content="광차를 찾지 못했습니다.")], ['{"success": false, "rationale": "x"}'], planner_replies
    )
    await service.start()
    try:
        thread = service.create_thread()
        record = await service.run(thread.thread_id, "광차로 가")
    finally:
        await service.stop()

    assert world.view_calls == [60.0, -60.0, 120.0, -120.0, 180.0] and world.locate_calls == []
    plan_prompt = service.planner_llm.prompts[-1][1].content[0]["text"]
    assert "'minecart' is not in the object list and was not found" in plan_prompt
    assert world.act_calls == [] and record.reply == "광차를 찾지 못했습니다."


@pytest.mark.parametrize("profile", ["local_qwen35_4b", "gpt41_mini"])
def test_profiles_load(profile):
    settings = load_settings(profile=profile, judge_profile=profile)
    assert settings.profile.name == profile
    assert settings.judge_profile.name == profile
    assert load_settings(profile=profile).judge_profile.name == "local_qwen35_4b_thinking"

