"""명령을 도구 호출 순서(계획)로 바꾸고, 세계 규칙으로 검사하고, 진행을 추적한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.dto import ActRequestDTO, ObservationDTO
from agent.prompts import INTENT_PROMPT, PLANNER_PROMPT, REWRITE_PROMPT
from agent.structured import structured_call
from agent.tools import format_call
from agent.world_text import describe_world, object_kind

# Unity CarrierAgent.inventoryCapacity와 같다(손 1개는 별도).
INVENTORY_CAPACITY = 4
# 요구의 목적지: 특정 객체 옆이 아니라 어디든 내려놓기
GROUND = "ground"


class PlanStep(BaseModel):
    action: Literal["move_to", "look_at", "pick", "store", "drop"]
    label: str | None = None
    near_label: str | None = None


class MotionGoal(BaseModel):
    action: Literal["move_to", "look_at"]
    label: str = Field(description="Exact target label; use the object name only if no matching label is known")


class CommandIntent(BaseModel):
    motion: list[MotionGoal] = Field(description="Explicit robot movement and gaze requests, in command order. Exclude travel implicit in pick/drop.")
    transport: bool = Field(description="True only if the user requests picking, storing, dropping or relocating objects")


class Requirement(BaseModel):
    object_type: str = Field(description="Object type exactly as shown after type= in the object list, e.g. coal")
    count: int = Field(ge=1, description="How many distinct objects of this type")
    destination: str | None = Field(
        description="Label of the object to place them next to, 'ground' to put them down anywhere, or null if they only need to be picked up"
    )

    def describe(self) -> str:
        if self.destination == GROUND:
            where = "put down"
        else:
            where = f"next to {self.destination}" if self.destination else "picked up"
        return f"{self.count} {self.object_type} {where}"


class PlanDraft(BaseModel):
    # 필드 순서가 곧 생각 순서다: 요구(개수 포함)를 먼저 정리하고 그다음 단계를 쓴다.
    requirements: list[Requirement] = Field(
        description="Object carrying/placement requirements only. MUST be [] when intent.transport is false. Robot movement and gaze are specified separately by intent.motion."
    )
    steps: list[PlanStep] = Field(description="Ordered tool calls from the current state, without finish")
    note: str = Field(default="", description="Why the command cannot be done, if it cannot")


@dataclass
class Plan:
    steps: list[ActRequestDTO]
    note: str = ""
    command: str = ""  # 계획에 쓴 (영어) 명령
    done: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.done:
            self.done = [False] * len(self.steps)

    def next_index(self) -> int | None:
        return next((i for i, finished in enumerate(self.done) if not finished), None)

    def remaining(self) -> list[str]:
        return [format_call(step) for step, finished in zip(self.steps, self.done) if not finished]

    def mark(self, request: ActRequestDTO) -> None:
        """성공한 도구 호출과 같은 첫 미완료 단계를 완료로 표시한다."""
        for i, (step, finished) in enumerate(zip(self.steps, self.done)):
            if not finished and step == request:
                self.done[i] = True
                return

    def summary(self) -> str:
        return " -> ".join(format_call(step) for step in self.steps) or f"(no actions) {self.note}"

    def render(self) -> str:
        if not self.steps:
            return f"[plan] No actions are needed or possible: {self.note or 'see the command'}. Call finish and tell the user."
        current = self.next_index()
        lines = ["[plan] made for this command. [x] = done, -> = do this next."]
        for i, step in enumerate(self.steps):
            mark = "[x]" if self.done[i] else ("->" if i == current else "[ ]")
            lines.append(f"{mark} {i + 1}. {format_call(step)}")
        if current is None:
            lines.append("All plan steps are done: call finish now.")
        return "\n".join(lines)


@dataclass
class Simulation:
    requests: list[ActRequestDTO]
    errors: list[str]
    holding: str | None
    inventory: list[str]
    placed: dict[str, str | None]  # 내려놓은 객체 -> near_label
    picked: set[str] = field(default_factory=set)  # 계획 중에 주운 객체


def simulate(steps: list[PlanStep], observation: ObservationDTO, near_radius: float = 3.0) -> Simulation:
    """계획을 손·인벤토리 규칙으로 따라가 본다. pick과 drop은 스스로 걸어가므로 위치는 따지지 않는다.

    고정된 객체를 줍거나, 처음부터 목적지 옆에 있던 객체를 다시 옮기는 낭비도 잡는다.
    """
    objects = {o.label: o for o in observation.objects}
    agent = observation.agent
    holding = agent.holding
    inventory = list(agent.inventory)
    start_carried = {agent.holding, *agent.inventory} - {None}

    def initially_near(label: str, anchor: str) -> bool:
        return label not in start_carried and objects[label].position.horizontal_distance(objects[anchor].position) <= near_radius
    picked_at: dict[str, int] = {}
    placed: dict[str, str | None] = {}

    def carried(label: str | None) -> bool:
        return label is not None and (label == holding or label in inventory)

    def since(label: str) -> str:
        return f" (picked at step {picked_at[label]})" if label in picked_at else " (carried from the start)"

    result: list[ActRequestDTO] = []
    errors: list[str] = []
    for number, step in enumerate(steps, 1):
        action = step.action
        label = None if action == "store" else (step.label or None)
        near = (step.near_label or None) if action == "drop" else None
        where = f"step {number} {format_call(ActRequestDTO(action=action, label=label, near_label=near))}"
        if action != "store" and label not in objects:
            errors.append(f"{where}: unknown label {label!r}; use a label from the object list")
            continue
        if near is not None and near not in objects:
            errors.append(f"{where}: unknown near_label {near!r}; use a label from the object list")
            continue

        if action in ("move_to", "look_at"):
            if carried(label):
                errors.append(f"{where}: {label} is carried at this point{since(label)}; carried objects cannot be targets")
                continue
        elif action == "pick":
            if not objects[label].pickable:
                errors.append(f"{where}: {label} is fixed and cannot be picked up")
                continue
            if carried(label):
                errors.append(f"{where}: {label} is already carried{since(label)}")
                continue
            if holding is not None:
                if len(inventory) >= INVENTORY_CAPACITY:
                    errors.append(f"{where}: hand and inventory are full; drop something before this pick")
                    continue
                inventory.append(holding)  # Unity가 손에 든 객체를 인벤토리로 옮긴다.
            holding = label
            picked_at[label] = number
            placed.pop(label, None)
        elif action == "store":
            if holding is None:
                errors.append(f"{where}: the hand is empty at this point; remove this store")
                continue
            inventory.append(holding)
            holding = None
        elif action == "drop":
            if not carried(label):
                errors.append(f"{where}: {label} is not carried at this point; pick it before dropping it")
                continue
            if near is not None and carried(near):
                errors.append(
                    f"{where}: {near} is carried at this point{since(near)}, so nothing can be placed next to it. "
                    f"Drop {label} next to {near} before picking {near}"
                )
                continue
            if near is not None and initially_near(label, near):
                errors.append(f"{where}: {label} was already next to {near} before this command; leave it where it is")
                continue
            if holding == label:
                holding = None
            else:
                inventory.remove(label)
            placed[label] = near
        result.append(ActRequestDTO(action=action, label=label, near_label=near))
    return Simulation(result, errors, holding, inventory, placed, set(picked_at))


def check_plan(steps: list[PlanStep], observation: ObservationDTO, near_radius: float = 3.0) -> tuple[list[ActRequestDTO], list[str]]:
    simulation = simulate(steps, observation, near_radius)
    return simulation.requests, simulation.errors


def drop_redundant_moves(requests: list[ActRequestDTO], explicit_moves: set[str] | None = None) -> list[ActRequestDTO]:
    """pick/drop의 보조 이동은 빼고, 사용자가 명시한 이동은 유지한다."""
    result = []
    explicit_moves = explicit_moves or set()
    for i, request in enumerate(requests):
        following = requests[i + 1] if i + 1 < len(requests) else None
        if (
            request.action == "move_to"
            and request.label not in explicit_moves
            and following is not None
            and (
                (following.action == "pick" and following.label == request.label)
                or (following.action == "drop" and following.near_label == request.label)
            )
        ):
            continue
        result.append(request)
    return result


def check_intent(intent: CommandIntent, draft: PlanDraft, requests: list[ActRequestDTO], observation: ObservationDTO) -> list[str]:
    """검사 중에도 원래 의도를 유지한다. 잘못 만든 운반 요구를 맞추려고 pick을 추가하지 않는다."""
    errors = []
    if not intent.transport:
        if draft.requirements:
            errors.append("This command requests no object transport. Remove the carrying/placement requirements; do not add pick/drop/store to satisfy them.")
        moves = {g.label for g in intent.motion if g.action == "move_to"}
        looks = {g.label for g in intent.motion}  # 이동 전 같은 대상을 바라보는 것은 허용한다.
        for request in requests:
            if (request.action not in ("move_to", "look_at")
                    or request.label not in (moves if request.action == "move_to" else looks)):
                errors.append(f"{format_call(request)} is not requested by the original movement/gaze intent. Remove it.")
    # 없는 대상은 행동 없이 이유를 설명할 수 있다. 알려진 fixed 대상은 건너뛰지 못한다.
    if any(observation.find(g.label) is None for g in intent.motion) and not requests and draft.note:
        return errors
    remaining = iter(requests)
    for goal in intent.motion:
        if not any(r.action == goal.action and r.label == goal.label for r in remaining):
            errors.append(f"Missing requested {goal.action}({goal.label}) in command order. Preserve the original movement/gaze intent.")
    return errors


def resolve_destination(destination: str | None, objects: dict) -> str | None:
    """목적지를 label이나 종류 이름으로 정리한다. 문장 안의 label을 찾고, 둘 다 없으면 '들고 있기'(None)로 본다."""
    if destination is None or destination in objects or destination == GROUND:
        return destination
    kinds = {object_kind(label) for label in objects}
    if destination in kinds:
        return destination
    mentioned = [label for label in objects if label in destination]
    if len(mentioned) == 1:
        return mentioned[0]
    mentioned_kinds = [kind for kind in kinds if kind in destination.split()]
    return mentioned_kinds[0] if len(mentioned_kinds) == 1 else None


def check_requirements(
    requirements: list[Requirement], simulation: Simulation, observation: ObservationDTO, near_radius: float
) -> list[str]:
    """모델이 정리한 요구와 계획의 최종 상태가 맞는지 본다. 개수 누락을 잡는다."""
    objects = {o.label: o for o in observation.objects}
    kinds = {object_kind(label) for label in objects}
    start_carried = {observation.agent.holding, *observation.agent.inventory} - {None}
    touched = {r.label for r in simulation.requests if r.action == "pick"} | set(simulation.placed)
    errors = []
    for requirement in requirements:
        if requirement.object_type not in kinds:
            errors.append(f"Unknown object type {requirement.object_type!r}; use a type from the object list, not an object label.")
            continue
        requirement = requirement.model_copy(update={"destination": resolve_destination(requirement.destination, objects)})
        kind = requirement.object_type
        if requirement.destination is None:
            # 주우라는 요구: 계획 중에 주웠거나 처음부터 들고 있던 객체를 센다.
            matched = sorted(label for label in simulation.picked | start_carried if object_kind(label) == kind)
        elif requirement.destination == GROUND:
            # 내려놓으라는 요구: 계획에서 내려놓은 객체를 센다.
            matched = sorted(label for label in simulation.placed if object_kind(label) == kind)
        else:
            # 목적지가 종류 이름이면 그 종류의 어느 객체 옆이든 된다.
            anchors = (
                [requirement.destination]
                if requirement.destination in objects
                else [label for label in objects if object_kind(label) == requirement.destination]
            )
            matched = sorted(
                label for label, near in simulation.placed.items() if near in anchors and object_kind(label) == kind
            )
            # 처음부터 그 옆에 있고 계획이 건드리지 않는 객체도 센다.
            matched += sorted(
                o.label
                for o in observation.objects
                if object_kind(o.label) == kind
                and o.label not in touched
                and o.label not in start_carried
                and any(o.position.horizontal_distance(objects[a].position) <= near_radius for a in anchors if a != o.label)
            )
        if len(matched) < requirement.count:
            errors.append(
                f"requirement '{requirement.describe()}' is not met: the plan ends with {len(matched)} "
                f"({', '.join(matched) or 'none'}). Add steps for {requirement.count - len(matched)} more distinct {kind}"
            )
    return errors


@dataclass
class PlanOutcome:
    plan: Plan | None
    errors: list[str]
    command: str  # 계획에 쓴 독립된 영어 지시
    drafts: list[str] = field(default_factory=list)  # 시도별 계획 원문(JSON), 디버깅용
    intent: CommandIntent | None = None


def carried_summary(observation: ObservationDTO) -> str:
    agent = observation.agent
    items = [f"{agent.holding} (in hand)"] if agent.holding else []
    items += [f"{label} (in inventory)" for label in agent.inventory]
    return ", ".join(items) or "nothing"


async def rewrite_command(
    llm: BaseChatModel,
    command: str,
    observation: ObservationDTO,
    history: str = "",
    type_names: dict[str, list[str]] | None = None,
) -> str:
    """명령을 대화 맥락과 현재 소지품으로 풀어서 독립된 영어 지시로 바꾼다.

    작은 모델은 한국어 명령의 대상과 '그거', '들고 있는 것' 같은 지시어를 자주 놓친다.
    """
    if not history and command.isascii():
        return command
    kinds = ", ".join(sorted({object_kind(o.label) for o in observation.objects}))
    system = f"{REWRITE_PROMPT}\nObject types in this world: {kinds}\nThe robot currently carries: {carried_summary(observation)}"
    system += "\nExact object labels: " + ", ".join(o.label for o in observation.objects)
    if type_names:
        words = "; ".join(f"{' / '.join(names)} = {kind}" for kind, names in type_names.items() if names)
        system += f"\nWords the user uses for these types: {words}"
    user = (f"Conversation so far:\n{history}\n\n" if history else "") + f"Latest command: {command}"
    response = await llm.ainvoke([SystemMessage(system), HumanMessage(content=user)])
    text = response.content if isinstance(response.content, str) else ""
    return text.strip().strip('"') or command


async def infer_intent(llm: BaseChatModel, command: str, observation: ObservationDTO, near_radius: float, history: str, notes: str) -> CommandIntent:
    """영어 재작성이나 플랜 초안과 독립적으로 원래 명령의 의도를 한 번 추출한다."""
    return await structured_call(llm, CommandIntent, [
        SystemMessage(INTENT_PROMPT),
        HumanMessage(f"Conversation:\n{history or '(none)'}\nOriginal user command: {command}\n\n{describe_world(observation, near_radius)}\n{notes}"),
    ])


async def make_plan(
    llm: BaseChatModel,
    command: str,
    observation: ObservationDTO,
    near_radius: float,
    failures: list[str] | None = None,
    history: str = "",
    rewritten: str | None = None,
    notes: str = "",
    attempts: int = 3,
    intent: CommandIntent | None = None,
    completed_motion: int = 0,
) -> PlanOutcome:
    english = rewritten or await rewrite_command(llm, command, observation, history)
    intent = intent or await infer_intent(llm, command, observation, near_radius, history, notes)
    remaining_intent = intent.model_copy(update={"motion": intent.motion[completed_motion:]})
    outcome = PlanOutcome(plan=None, errors=[], command=english, intent=intent)
    if not intent.transport and intent.motion and all(observation.find(g.label) for g in remaining_intent.motion):
        # 모델이 추출한 이동/시선은 이미 완전한 도구 인자다. 운반 계획을 다시 생성하지 않는다.
        draft = PlanDraft(requirements=[], steps=[PlanStep(action=g.action, label=g.label) for g in remaining_intent.motion])
        simulation = simulate(draft.steps, observation, near_radius)
        outcome.errors = simulation.errors
        if not outcome.errors:
            outcome.plan = Plan(steps=simulation.requests, command=english)
        return outcome
    text = (f"Command: {english}\nOriginal user command: {command}\n"
            f"Fixed intent (remaining goals from the original command): {remaining_intent.model_dump_json()}\n\n"
            f"Current state:\n{describe_world(observation, near_radius)}")
    if notes:
        text += f"\n{notes}"
    if failures:
        text += "\n\nThe previous plan failed. Recent failed actions:\n" + "\n".join(f"- {f}" for f in failures[-5:])
        text += "\nPlan the remaining work from the current state."
    messages: list[BaseMessage] = [
        SystemMessage(PLANNER_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{observation.image_png_base64}"}},
            ]
        ),
    ]
    for _ in range(attempts):
        draft = await structured_call(llm, PlanDraft, messages)
        outcome.drafts.append(draft.model_dump_json())
        simulation = simulate(draft.steps, observation, near_radius)
        goal = "; ".join(r.describe() for r in draft.requirements)
        requests = drop_redundant_moves(simulation.requests, {g.label for g in remaining_intent.motion if g.action == "move_to"})
        plan = Plan(steps=requests, note=draft.note or goal, command=english)
        # 의도 오류부터 고친다. 잘못된 운반 요구를 만족시키도록 행동을 늘리지 않는다.
        outcome.errors = (
            check_intent(remaining_intent, draft, requests, observation)
            or simulation.errors
            or check_requirements(draft.requirements, simulation, observation, near_radius)
        )
        if not outcome.errors:
            outcome.plan = plan
            return outcome
        messages.append(AIMessage(content=draft.model_dump_json()))
        messages.append(
            HumanMessage(
                content="This plan is wrong:\n" + "\n".join(f"- {e}" for e in outcome.errors)
                + "\nRe-read the original command and fixed intent. Fix incorrect requirements as well as steps; do not invent object transport. Return a corrected complete plan from the current state."
            )
        )
    return outcome
