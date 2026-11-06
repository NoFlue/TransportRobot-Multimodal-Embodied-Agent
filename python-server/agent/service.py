"""Thread(대화)와 명령 대기열을 관리하고 ORA 그래프와 판정을 실행한다."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.config import Settings
from agent.graph import CommandContext, StepRecord, build_graph, unanswered_tool_calls
from agent.judge import judge_command
from agent.llm import create_chat_model
from agent.metrics import RunRecorder
from agent.unity_client import UnityClient

Listener = Callable[[str, dict], None]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class CommandRecord:
    command_id: str
    thread_id: str
    text: str
    status: str = "queued"  # queued -> running -> judging -> done | failed | cancelled
    created_at: str = field(default_factory=_now)
    received_at: float = field(default_factory=time.perf_counter)
    reply: str = ""
    end_reason: str | None = None
    steps: int = 0
    tool_success_rate: float | None = None
    command_to_first_action_seconds: float | None = None
    planning_seconds: float | None = None
    total_seconds: float | None = None
    judge_success: bool | None = None
    judge_rationale: str = ""
    error: str | None = None
    done: asyncio.Future | None = field(default=None, repr=False)

    def view(self) -> dict:
        return {
            "command_id": self.command_id,
            "text": self.text,
            "status": self.status,
            "created_at": self.created_at,
            "reply": self.reply,
            "end_reason": self.end_reason,
            "steps": self.steps,
            "tool_success_rate": self.tool_success_rate,
            "command_to_first_action_seconds": self.command_to_first_action_seconds,
            "planning_seconds": self.planning_seconds,
            "total_seconds": self.total_seconds,
            "judge_success": self.judge_success,
            "judge_rationale": self.judge_rationale,
            "error": self.error,
        }


@dataclass
class ThreadRecord:
    thread_id: str
    created_at: str = field(default_factory=_now)
    messages: list[dict] = field(default_factory=list)
    commands: list[CommandRecord] = field(default_factory=list)

    def view(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "created_at": self.created_at,
            "messages": self.messages,
            "commands": [command.view() for command in self.commands],
        }


class AgentService:
    def __init__(
        self,
        settings: Settings,
        *,
        unity: UnityClient | None = None,
        llm: BaseChatModel | None = None,
        judge_llm: BaseChatModel | None = None,
        planner_llm: BaseChatModel | None = None,
        recorder: RunRecorder | None = None,
    ):
        self.settings = settings
        agent_config = settings.app.agent
        self.unity = unity or UnityClient(settings.app.unity)
        self.llm = llm or create_chat_model(settings, settings.profile)
        if judge_llm is None:
            same = settings.judge_profile.name == settings.profile.name
            judge_llm = self.llm if same else create_chat_model(settings, settings.judge_profile)
        self.judge_llm = judge_llm
        if planner_llm is None and settings.planner_profile is not None:
            planner = settings.planner_profile
            if planner.name == settings.profile.name:
                planner_llm = self.llm
            elif planner.name == settings.judge_profile.name:
                planner_llm = judge_llm
            else:
                planner_llm = create_chat_model(settings, planner)
        self.planner_llm = planner_llm
        runs_dir = Path(settings.app.logging.runs_dir)
        if not runs_dir.is_absolute():
            runs_dir = settings.config_dir.parent / runs_dir
        self.recorder = recorder or RunRecorder(runs_dir, settings.app.logging.save_images)
        self.graph = build_graph(
            self.llm,
            self.unity,
            near_radius=agent_config.near_radius_meters,
            max_steps=agent_config.max_steps,
            replan_after_failures=agent_config.replan_after_failures,
            max_consecutive_failures=agent_config.max_consecutive_failures,
            planner_llm=self.planner_llm,
            ground_unknown_objects=agent_config.ground_unknown_objects,
            type_names=agent_config.type_names,
            checkpointer=InMemorySaver(),
        )
        self.threads: dict[str, ThreadRecord] = {}
        self.commands: dict[str, CommandRecord] = {}
        self.listeners: list[Listener] = []
        self._queue: asyncio.Queue[CommandRecord] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._running: tuple[CommandRecord, asyncio.Task] | None = None

    # ---------------- 수명 ----------------

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._work())

    async def stop(self) -> None:
        if self._running is not None:
            self._running[1].cancel()
            await asyncio.gather(self._running[1], return_exceptions=True)
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
        await self.unity.aclose()

    # ---------------- 공개 API ----------------

    def create_thread(self) -> ThreadRecord:
        thread = ThreadRecord(thread_id=str(uuid.uuid4()))
        self.threads[thread.thread_id] = thread
        return thread

    def submit(self, thread_id: str, text: str) -> CommandRecord:
        if thread_id not in self.threads:
            raise KeyError(thread_id)
        record = CommandRecord(command_id=str(uuid.uuid4()), thread_id=thread_id, text=text.strip())
        record.done = asyncio.get_running_loop().create_future()
        self.threads[thread_id].commands.append(record)
        self.commands[record.command_id] = record
        self._post(thread_id, "user", record.text, record.command_id)
        self._queue.put_nowait(record)
        return record

    async def run(self, thread_id: str, text: str) -> CommandRecord:
        record = self.submit(thread_id, text)
        assert record.done is not None
        return await record.done

    async def cancel(self, command_id: str) -> bool:
        record = self.commands.get(command_id)
        if record is None:
            raise KeyError(command_id)
        if record.status == "queued":
            self._finish(record, "cancelled", "cancelled")
            return True
        if self._running and self._running[0] is record:
            self._running[1].cancel()
            return True
        return False

    # ---------------- 실행 ----------------

    def _post(self, thread_id: str, role: str, text: str, command_id: str | None = None) -> None:
        message = {"role": role, "text": text, "command_id": command_id, "time": _now()}
        self.threads[thread_id].messages.append(message)
        for listener in self.listeners:
            listener(thread_id, message)

    async def _work(self) -> None:
        while True:
            record = await self._queue.get()
            if record.status != "queued":
                continue
            task = asyncio.create_task(self._execute(record))
            self._running = (record, task)
            try:
                await asyncio.wait({task})
            finally:
                self._running = None
            if task.cancelled() and not record.done.done():
                self._finish(record, "cancelled", "cancelled")

    async def _execute(self, record: CommandRecord) -> None:
        agent_config = self.settings.app.agent
        record.status = "running"
        thread = self.threads[record.thread_id]
        earlier = [m for m in thread.messages if m["command_id"] != record.command_id and m["role"] in ("user", "agent")]
        history = "\n".join(f"{'User' if m['role'] == 'user' else 'Robot'}: {m['text']}" for m in earlier[-8:])
        ctx = CommandContext(
            command_id=record.command_id,
            thread_id=record.thread_id,
            text=record.text,
            received_at=record.received_at,
            history=history,
            on_event=lambda role, text: self._post(record.thread_id, role, text, record.command_id),
            on_step=lambda step: self._record_step(record, step),
            on_plan=lambda plan: self.recorder.record_plan(record.thread_id, record.command_id, plan),
        )
        config = {"configurable": {"thread_id": record.thread_id}, "recursion_limit": agent_config.max_steps * 3 + 10}
        status = "done"
        try:
            await self.unity.sync_clock()
            previous = await self.graph.aget_state(config)
            repairs = unanswered_tool_calls(previous.values.get("messages", []))
            graph_input = {
                "messages": [*repairs, HumanMessage(content=record.text)],
                "pending_observation": None,
                "steps": 0,
                "end_reason": None,
            }
            final = await asyncio.wait_for(
                self.graph.ainvoke(graph_input, config, context=ctx), timeout=agent_config.max_command_seconds
            )
            record.end_reason = final.get("end_reason") or "answered"
        except asyncio.TimeoutError:
            record.end_reason = "timeout"
            self._post(record.thread_id, "system", "Command timed out.", record.command_id)
            await self._stop_unity_action()
        except asyncio.CancelledError:
            record.end_reason = "cancelled"
            status = "cancelled"
            self._post(record.thread_id, "system", "Command cancelled.", record.command_id)
            await self._stop_unity_action()
        except Exception as exc:  # noqa: BLE001 - 실패 이유를 대화와 기록에 남긴다.
            record.end_reason = "error"
            record.error = f"{type(exc).__name__}: {exc}"
            status = "failed"
            self._post(record.thread_id, "system", f"Error: {record.error}", record.command_id)

        record.reply = ctx.reply
        record.steps = len(ctx.steps)
        record.planning_seconds = round(ctx.planning_seconds, 3) if self.planner_llm is not None else None
        if ctx.steps:
            record.tool_success_rate = sum(step.tool_success for step in ctx.steps) / len(ctx.steps)
        if ctx.first_action_at is not None:
            record.command_to_first_action_seconds = ctx.first_action_at - record.received_at

        if status != "cancelled" and self.settings.app.judge.enabled:
            await self._judge(record, ctx)
        self._finish(record, status, record.end_reason)

    async def _judge(self, record: CommandRecord, ctx: CommandContext) -> None:
        record.status = "judging"
        try:
            observation = await self.unity.observe()
        except Exception:  # noqa: BLE001 - Unity가 끊겼으면 마지막 관측으로 판정한다.
            observation = ctx.last_observation
        if observation is None:
            return
        try:
            verdict = await judge_command(
                self.judge_llm,
                record.text,
                observation,
                self.settings.app.agent.near_radius_meters,
                initial=ctx.first_observation,
            )
        except Exception as exc:  # noqa: BLE001
            record.judge_rationale = f"judge error: {type(exc).__name__}: {exc}"
            self._post(record.thread_id, "system", f"Judge failed: {exc}", record.command_id)
            return
        record.judge_success = verdict.success
        record.judge_rationale = verdict.rationale
        outcome = "success" if verdict.success else "failure"
        self._post(record.thread_id, "system", f"Judge: {outcome} - {verdict.rationale}", record.command_id)

    async def _stop_unity_action(self) -> None:
        try:
            await self.unity.cancel()
        except Exception:  # noqa: BLE001 - 정지 요청 실패는 기록만 남긴다.
            pass

    def _record_step(self, record: CommandRecord, step: StepRecord) -> None:
        self.recorder.record_step(record.thread_id, record.command_id, step)

    def _finish(self, record: CommandRecord, status: str, end_reason: str | None) -> None:
        record.status = status
        record.end_reason = end_reason
        record.total_seconds = time.perf_counter() - record.received_at
        self.recorder.record_result(
            {
                "thread_id": record.thread_id,
                "command_id": record.command_id,
                "text": record.text,
                "model_profile": self.settings.profile.name,
                "judge_profile": self.settings.judge_profile.name,
                "planner_profile": self.settings.planner_profile.name if self.settings.planner_profile else "",
                "planning_seconds": record.planning_seconds,
                "command_to_first_action_seconds": record.command_to_first_action_seconds,
                "judge_success": record.judge_success,
                "judge_rationale": record.judge_rationale,
                "steps": record.steps,
                "tool_success_rate": record.tool_success_rate,
                "end_reason": end_reason,
                "total_seconds": round(record.total_seconds, 3),
            }
        )
        if record.done is not None and not record.done.done():
            record.done.set_result(record)
