"""VLM에 bind하는 도구 스키마. 각 도구는 Unity AgentAction 하나에 대응한다."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from agent.dto import ActRequestDTO


class move_to(BaseModel):
    """Walk to stand next to the object with this label. Works for any label in the object list, even if it is not visible in the image."""

    label: str = Field(description="Exact object label, e.g. coal_1")


class look_at(BaseModel):
    """Turn your body and camera to face the object with this label."""

    label: str = Field(description="Exact object label")


class pick(BaseModel):
    """Walk to the object with this label and pick it up. If your hand is full, the object in your hand goes to the inventory automatically."""

    label: str = Field(description="Exact label of the object to pick up")


class store(BaseModel):
    """Put the object in your hand into your inventory. Rarely needed: pick does this automatically."""


class drop(BaseModel):
    """Put down an object you carry (in hand or inventory). With near_label, walk next to that object and place it there."""

    label: str = Field(description="Exact label of the carried object to put down")
    near_label: str | None = Field(default=None, description="Label of the object to place it next to, e.g. furnace_1")


class finish(BaseModel):
    """Call this once the user's command is complete, or when it cannot be completed. It ends your work on the command and sends the summary to the user."""

    summary: str = Field(description="Short report for the user, in the user's language")


FINISH_TOOL = finish.__name__
# finish는 Unity 행동이 아니라 명령 처리를 끝내는 도구다.
AGENT_TOOLS: list[type[BaseModel]] = [move_to, look_at, pick, store, drop, finish]
_TOOLS_BY_NAME = {tool.__name__: tool for tool in AGENT_TOOLS if tool is not finish}


class ToolCallError(ValueError):
    pass


def to_act_request(name: str, args: dict | None) -> ActRequestDTO:
    """VLM의 tool call을 Unity /act 요청으로 바꾼다. 잘못된 호출은 ToolCallError."""
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        raise ToolCallError(f"Unknown tool '{name}'. Available tools: {', '.join(_TOOLS_BY_NAME)}.")
    try:
        parsed = tool.model_validate(args or {})
    except ValidationError as exc:
        raise ToolCallError(f"Invalid arguments for {name}: {exc.errors(include_url=False)}") from exc
    return ActRequestDTO(action=name, **parsed.model_dump())


def format_call(request: ActRequestDTO) -> str:
    args = [value for value in (request.label, request.near_label and f"near {request.near_label}") if value]
    return f"{request.action}({', '.join(args)})"
