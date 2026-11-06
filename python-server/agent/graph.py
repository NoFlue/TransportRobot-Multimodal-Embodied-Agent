"""LangGraph ORA(Observation -> Reasoning -> Act) 루프.

계획이 있으면 계획 단계를 그대로 실행하고(계획 주도), 단계가 실패하면 현재 상태에서 다시 계획한다.
계획이 없거나 다시 만든 계획의 단계도 실패하면 실행 모델이 한 단계씩 고른다(반응형).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from agent.dto import ObservationDTO
from agent.grounding import ground_objects
from agent.planner import Plan, PlanOutcome, make_plan, rewrite_command
from agent.prompts import SUMMARY_PROMPT, SYSTEM_PROMPT
from agent.tools import AGENT_TOOLS, FINISH_TOOL, ToolCallError, format_call, to_act_request
from agent.unity_client import UnityClient
from agent.world_text import describe_world

# 같은 호출이 이 횟수만큼 연달아 나오면 Unity를 부르지 않고 되돌려 보낸다.
MAX_SAME_CALLS = 2
STUCK_REPLY = "연속 {count}번 행동이 실패해서 명령을 중단했습니다. 마지막 오류: {error}"


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # act가 받은 실행 후 관측. perceive가 메시지로 바꾸고 비운다.
    pending_observation: dict | None
    # 이미지가 들어 있는 최신 관측 메시지 id. 다음 관측 때 텍스트만 남긴다.
    observation_message_id: str | None
    steps: int
    end_reason: str | None


@dataclass
class StepRecord:
    step: int
    tool_name: str
    tool_args: dict
    reasoning_seconds: float
    step_to_call_seconds: float | None
    execution_seconds: float | None
    tool_success: bool
    feedback: str
    image_png_base64: str | None = None


@dataclass
class CommandContext:
    """명령 하나를 처리하는 동안 노드가 공유하는 실행 정보와 측정값."""

    command_id: str
    thread_id: str
    text: str
    received_at: float
    on_event: Callable[[str, str], None] = lambda role, text: None
    on_step: Callable[[StepRecord], None] = lambda record: None
    on_plan: Callable[[dict], None] = lambda record: None
    # 같은 대화의 이전 명령과 응답. 지시어를 풀 때 쓴다.
    history: str = ""
    # 계획에 쓰는 독립된 영어 지시와, 관측 이미지로 찾은 객체 메모
    rewritten: str | None = None
    grounding_notes: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    first_observation: ObservationDTO | None = None
    last_observation: ObservationDTO | None = None
    first_action_at: float | None = None
    step_started_at: float = 0.0
    reasoning_seconds: float = 0.0
    reply: str = ""
    last_call: tuple | None = None
    same_call_count: int = 0
    # 연속 실패 기록. 성공하면 비운다.
    recent_failures: list[str] = field(default_factory=list)
    plan: Plan | None = None
    planned: bool = False
    replanned_this_streak: bool = False
    plan_made_at: int = 0  # 계획을 만든 시점까지 실행한 단계 수
    planning_seconds: float = 0.0
    finish_rejections: int = 0

    def add_step(self, record: StepRecord) -> None:
        self.steps.append(record)
        if record.tool_success:
            self.recent_failures.clear()
            self.replanned_this_streak = False
        else:
            self.recent_failures.append(f"{record.tool_name}({_format_args(record.tool_args)}) -> {record.feedback}")
        self.on_step(record)


def _format_args(args: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in args.items() if value is not None)


def text_of(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")


def compact_observation(message: AnyMessage) -> HumanMessage:
    """지난 관측은 머리줄만 남긴다. 현재 상태는 최신 관측에만 있다."""
    header = text_of(message).split("\n", 1)[0]
    return HumanMessage(id=message.id, content=f"{header} (details omitted; see the latest observation)")


def recovery_note(observation: ObservationDTO, failures: list[str]) -> str:
    """연속 실패 뒤에 관측에 붙이는 재계획 지시."""
    agent = observation.agent
    carried = [f"{agent.holding} (in hand)"] if agent.holding else []
    carried += [f"{label} (in inventory)" for label in agent.inventory]
    lines = [
        f"[recovery] Your last {len(failures)} actions failed:",
        *(f"- {failure}" for failure in failures[-5:]),
        f"You are carrying: {', '.join(carried) or 'nothing'}.",
        "Objects you carry move with you: you cannot move_to them, look_at them, or use them as near_label.",
        "To place something next to a carried object, first put that object on the ground with drop(label=...).",
        "Do not repeat a failed call. Read the error, choose a different action that changes the situation,",
        "or call finish to tell the user why the command cannot be completed.",
    ]
    return "\n".join(lines)


def observation_message(
    observation: ObservationDTO, near_radius: float, step: int, note: str | None = None
) -> HumanMessage:
    header = "Current observation" if step == 0 else f"Observation after step {step}"
    text = f"{header} (image: your first-person camera view)\n{describe_world(observation, near_radius)}"
    if note:
        text = f"{text}\n{note}"
    return HumanMessage(
        id=str(uuid.uuid4()),
        content=[
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{observation.image_png_base64}"}},
        ],
    )


def unanswered_tool_calls(messages: list[AnyMessage]) -> list[ToolMessage]:
    """중단된 명령이 남긴 응답 없는 tool call을 닫는 메시지를 만든다."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    repairs = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["id"] not in answered:
                    repairs.append(
                        ToolMessage(content="failed: interrupted before execution", tool_call_id=call["id"], name=call["name"])
                    )
    return repairs


def follows_plan(ctx: CommandContext) -> bool:
    """계획 단계를 그대로 실행할지. 실패한 뒤에는 그 실패 이후에 만든 계획일 때만 따른다."""
    if ctx.plan is None:
        return False
    return not ctx.recent_failures or ctx.plan_made_at >= len(ctx.steps)


def plan_call(request) -> AIMessage:
    args = request.model_dump(exclude_none=True, exclude={"action"})
    return AIMessage(
        content=f"Following the plan: {format_call(request)}",
        tool_calls=[{"name": request.action, "args": args, "id": f"plan_{uuid.uuid4().hex[:12]}"}],
    )


def build_graph(
    llm: BaseChatModel,
    unity: UnityClient,
    *,
    near_radius: float,
    max_steps: int,
    replan_after_failures: int = 3,
    max_consecutive_failures: int = 6,
    planner_llm: BaseChatModel | None = None,
    ground_unknown_objects: bool = True,
    type_names: dict[str, list[str]] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    model = llm.bind_tools(AGENT_TOOLS)

    async def ground(ctx: CommandContext, observation: ObservationDTO) -> ObservationDTO:
        """명령을 다시 쓰고, 목록에 없는 객체를 시점 이미지에서 찾아 Unity에 등록한다."""
        started = time.perf_counter()
        try:
            ctx.rewritten = await rewrite_command(planner_llm, ctx.text, observation, ctx.history, type_names)
        except Exception as exc:  # noqa: BLE001 - 다시 쓰기 실패는 계획 단계가 다시 시도한다.
            ctx.on_event("system", f"Command rewrite failed: {type(exc).__name__}: {exc}")
            return observation
        if not ground_unknown_objects:
            return observation
        try:
            report = await ground_objects(planner_llm, unity, ctx.rewritten, observation)
        except Exception as exc:  # noqa: BLE001 - 찾지 못해도 아는 객체로 계획은 한다.
            ctx.on_event("system", f"Object grounding failed: {type(exc).__name__}: {exc}")
            return observation
        seconds = time.perf_counter() - started
        ctx.planning_seconds += seconds
        ctx.grounding_notes = report.notes()
        ctx.on_plan(
            {"stage": "grounding", "command": ctx.text, "rewritten": ctx.rewritten,
             "objects": report.as_record(), "seconds": round(seconds, 3)}
        )
        for item in report.items:
            if item.status == "found":
                ctx.on_event("system", f"'{item.name}' was not in the object list; found it by looking around ({item.yaw:+.0f}°): {item.label}")
            elif item.status == "not_found":
                ctx.on_event("system", f"'{item.name}' is not in the object list and was not found by looking around.")
        if report.registered:
            # 새로 등록한 객체가 들어간 관측으로 바꾼다. 아무것도 움직이지 않았으므로 시작 상태로도 쓴다.
            observation = await unity.observe()
            ctx.first_observation = observation
            ctx.last_observation = observation
        return observation

    async def plan(ctx: CommandContext, observation: ObservationDTO, failures: list[str] | None) -> None:
        started = time.perf_counter()
        try:
            outcome = await make_plan(
                planner_llm, ctx.text, observation, near_radius, failures, ctx.history,
                rewritten=ctx.rewritten, notes=ctx.grounding_notes,
            )
        except Exception as exc:  # noqa: BLE001 - 계획 없이도 실행은 계속한다.
            outcome = PlanOutcome(plan=None, errors=[f"{type(exc).__name__}: {exc}"], command=ctx.text)
        seconds = time.perf_counter() - started
        ctx.planning_seconds += seconds
        ctx.on_plan(
            {
                "stage": "plan",
                "replan": bool(failures),
                "command": ctx.text,
                "rewritten": outcome.command,
                "plan": outcome.plan.summary() if outcome.plan else None,
                "errors": outcome.errors,
                "drafts": outcome.drafts,
                "seconds": round(seconds, 3),
            }
        )
        if outcome.plan is not None:
            ctx.plan = outcome.plan
            ctx.plan_made_at = len(ctx.steps)
            ctx.on_event("plan", f"{'Re-plan' if failures else 'Plan'} [{outcome.command}]: {outcome.plan.summary()}")
            if outcome.errors:
                ctx.on_event("system", "Plan may not match the command: " + "; ".join(outcome.errors[:2]))
        else:
            if failures:
                ctx.plan = None  # 다시 계획하지 못하면 실행 모델이 이어받는다.
            ctx.on_event("system", "Planner could not make a valid plan: " + "; ".join(outcome.errors[:3]))

    async def summarize(ctx: CommandContext) -> str:
        actions = [f"{s.tool_name}({_format_args(s.tool_args)}) -> {'ok' if s.tool_success else 'failed'}" for s in ctx.steps]
        note = ctx.plan.note if ctx.plan is not None else ""
        text = f"User command: {ctx.text}\nActions: {'; '.join(actions) or 'none'}\nPlanner note: {note or '-'}"
        if ctx.last_observation is not None:
            text += f"\n\nCurrent state:\n{describe_world(ctx.last_observation, near_radius)}"
        try:
            reply = text_of(await llm.ainvoke([SystemMessage(SUMMARY_PROMPT), HumanMessage(content=text)])).strip()
        except Exception:  # noqa: BLE001 - 요약 실패는 기본 문장으로 대신한다.
            reply = ""
        return reply or note or "Done."

    async def perceive(state: AgentState, runtime: Runtime[CommandContext]) -> dict[str, Any]:
        ctx = runtime.context
        pending = state.get("pending_observation")
        observation = ObservationDTO.model_validate(pending) if pending else await unity.observe()
        ctx.last_observation = observation
        if ctx.first_observation is None:
            ctx.first_observation = observation
        if planner_llm is not None and not ctx.planned:
            ctx.planned = True
            observation = await ground(ctx, observation)
            await plan(ctx, observation, None)
        elif (
            planner_llm is not None
            and ctx.recent_failures
            and not ctx.replanned_this_streak
            and (ctx.plan is not None or len(ctx.recent_failures) >= replan_after_failures)
        ):
            # 계획을 따르다 실패하면 곧바로, 반응형으로 실패가 쌓이면 다시 계획한다.
            ctx.replanned_this_streak = True
            await plan(ctx, observation, list(ctx.recent_failures))

        updates: list[AnyMessage] = []
        previous_id = state.get("observation_message_id")
        if previous_id:
            for message in state.get("messages", []):
                if message.id == previous_id:
                    updates.append(compact_observation(message))
        notes = [ctx.plan.render()] if ctx.plan is not None else []
        if len(ctx.recent_failures) >= replan_after_failures:
            notes.append(recovery_note(observation, ctx.recent_failures))
            ctx.on_event("system", f"{len(ctx.recent_failures)} actions failed in a row; asking the agent to re-plan.")
        message = observation_message(observation, near_radius, state.get("steps", 0), "\n".join(notes) or None)
        return {"messages": [*updates, message], "pending_observation": None, "observation_message_id": message.id}

    async def reason(state: AgentState, runtime: Runtime[CommandContext]) -> dict[str, Any]:
        ctx = runtime.context
        if state.get("steps", 0) >= max_steps:
            ctx.on_event("system", f"Step limit ({max_steps}) reached.")
            return {"end_reason": "max_steps"}
        if len(ctx.recent_failures) >= max_consecutive_failures:
            last_error = ctx.recent_failures[-1].split(" -> ", 1)[-1]
            ctx.reply = STUCK_REPLY.format(count=len(ctx.recent_failures), error=last_error)
            ctx.on_event("agent", ctx.reply)
            return {"messages": [AIMessage(content=ctx.reply)], "end_reason": "stuck"}

        ctx.step_started_at = time.perf_counter()
        if follows_plan(ctx):
            index = ctx.plan.next_index()
            if index is None:
                ctx.reply = await summarize(ctx)
                ctx.on_event("agent", ctx.reply)
                return {"messages": [AIMessage(content=ctx.reply)], "end_reason": "answered"}
            ctx.reasoning_seconds = 0.0
            return {"messages": [plan_call(ctx.plan.steps[index])]}

        response = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        ctx.reasoning_seconds = time.perf_counter() - ctx.step_started_at

        thought = text_of(response).strip()
        finish_call = next((c for c in response.tool_calls if c["name"] == FINISH_TOOL), None)
        remaining = ctx.plan.remaining() if ctx.plan is not None else []
        if finish_call is not None and remaining and ctx.finish_rejections < 1:
            # 계획이 남았는데 끝내려 하면 한 번 되돌린다. 두 번째 finish는 받아들인다.
            ctx.finish_rejections += 1
            feedback = (
                f"failed: the plan still has steps left: {', '.join(remaining)}. "
                "Do them, or call finish again if they cannot be done."
            )
            ctx.on_event("system", "Finish rejected: plan steps remain.")
            closing = [
                ToolMessage(content=feedback if c is finish_call else "failed: ignored", tool_call_id=c["id"], name=c["name"])
                for c in response.tool_calls
            ]
            pending = ctx.last_observation.model_dump() if ctx.last_observation else None
            return {"messages": [response, *closing], "pending_observation": pending}
        if finish_call is not None:
            ctx.reply = str(finish_call["args"].get("summary", "")).strip() or thought
            ctx.on_event("agent", ctx.reply or "(empty reply)")
            closing = [
                ToolMessage(content="done" if c is finish_call else "failed: ignored, the command was finished", tool_call_id=c["id"], name=c["name"])
                for c in response.tool_calls
            ]
            return {"messages": [response, *closing], "end_reason": "answered"}
        if response.tool_calls:
            if thought:
                ctx.on_event("thought", thought)
            return {"messages": [response]}
        ctx.reply = thought
        ctx.on_event("agent", thought or "(empty reply)")
        return {"messages": [response], "end_reason": "answered"}

    async def act(state: AgentState, runtime: Runtime[CommandContext]) -> dict[str, Any]:
        ctx = runtime.context
        response = state["messages"][-1]
        call, *extra = response.tool_calls
        skipped = [
            ToolMessage(content="failed: ignored, call only one tool per turn", tool_call_id=c["id"], name=c["name"])
            for c in extra
        ]
        step = state.get("steps", 0) + 1
        seen_image = ctx.last_observation.image_png_base64 if ctx.last_observation else None

        try:
            request = to_act_request(call["name"], call["args"])
        except ToolCallError as exc:
            ctx.add_step(StepRecord(step, call["name"], call["args"], ctx.reasoning_seconds, None, None, False, str(exc), seen_image))
            ctx.on_event("action", f"{call['name']} -> invalid call: {exc}")
            reply = ToolMessage(content=f"failed: {exc}", tool_call_id=call["id"], name=call["name"])
            pending = ctx.last_observation.model_dump() if ctx.last_observation else None
            return {"messages": [reply, *skipped], "steps": step, "pending_observation": pending}

        key = (request.action, request.label, request.near_label)
        ctx.same_call_count = ctx.same_call_count + 1 if key == ctx.last_call else 1
        ctx.last_call = key
        if ctx.same_call_count > MAX_SAME_CALLS:
            feedback = (
                f"Not executed: you already called {format_call(request)} {MAX_SAME_CALLS} times in a row. "
                "If the command is complete, call finish; otherwise choose a different action."
            )
            ctx.add_step(StepRecord(step, request.action, call["args"], ctx.reasoning_seconds, None, None, False, feedback, seen_image))
            ctx.on_event("action", f"{format_call(request)} -> {feedback}")
            reply = ToolMessage(content=f"failed: {feedback}", tool_call_id=call["id"], name=call["name"])
            pending = ctx.last_observation.model_dump() if ctx.last_observation else None
            return {"messages": [reply, *skipped], "steps": step, "pending_observation": pending}

        result = await unity.act(request)
        started = unity.to_local_time(result.started_at)
        if ctx.first_action_at is None:
            ctx.first_action_at = started
        ctx.add_step(
            StepRecord(
                step=step,
                tool_name=request.action,
                tool_args=request.model_dump(exclude_none=True, exclude={"action"}),
                reasoning_seconds=ctx.reasoning_seconds,
                step_to_call_seconds=started - ctx.step_started_at,
                execution_seconds=result.finished_at - result.started_at,
                tool_success=result.success,
                feedback=result.feedback,
                image_png_base64=seen_image,
            )
        )
        outcome = "success" if result.success else "failed"
        if result.success and ctx.plan is not None:
            ctx.plan.mark(request)
        ctx.on_event("action", f"{format_call(request)} -> {outcome}: {result.feedback}")
        reply = ToolMessage(content=f"{outcome}: {result.feedback}", tool_call_id=call["id"], name=call["name"])
        return {"messages": [reply, *skipped], "steps": step, "pending_observation": result.observation.model_dump()}

    def after_reason(state: AgentState) -> str:
        if state.get("end_reason"):
            return END
        last = state["messages"][-1]
        return "act" if isinstance(last, AIMessage) and last.tool_calls else "perceive"

    graph = StateGraph(AgentState, context_schema=CommandContext)
    graph.add_node("perceive", perceive)
    graph.add_node("reason", reason)
    graph.add_node("act", act)
    graph.add_edge(START, "perceive")
    graph.add_edge("perceive", "reason")
    graph.add_conditional_edges("reason", after_reason, ["act", "perceive", END])
    graph.add_edge("act", "perceive")
    return graph.compile(checkpointer=checkpointer)

