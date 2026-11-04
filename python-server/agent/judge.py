"""명령 처리가 끝난 뒤 최종 관측을 보고 성공 여부를 판정한다."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agent.dto import ObservationDTO
from agent.prompts import JUDGE_PROMPT
from agent.structured import structured_call
from agent.world_text import describe_world


class JudgeVerdict(BaseModel):
    success: bool
    rationale: str = ""


async def judge_command(
    llm: BaseChatModel,
    command: str,
    observation: ObservationDTO,
    near_radius: float,
    initial: ObservationDTO | None = None,
) -> JudgeVerdict:
    before = describe_world(initial, near_radius) if initial else "(not available)"
    content = [
        {
            "type": "text",
            "text": (
                f"Command: {command}\n\nInitial world state:\n{before}\n\n"
                f"Final world state:\n{describe_world(observation, near_radius)}"
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{observation.image_png_base64}"}},
    ]
    messages = [SystemMessage(JUDGE_PROMPT), HumanMessage(content=content)]
    return await structured_call(llm, JudgeVerdict, messages)
