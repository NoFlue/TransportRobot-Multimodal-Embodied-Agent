"""명령이 가리키는 객체를 관측과 맞춘다(grounding).

1. 명령에 나온 객체를 이미 ObjectData로 받은 것(known)과 처음 보는 것(unknown)으로 나눈다.
2. 처음 보는 객체는 고개를 돌린 시점 이미지에서 VLM으로 찾고 이미지 좌표를 예측한다.
3. Unity /perceive/locate가 그 좌표로 객체를 특정해 label을 붙여 등록하고 ObjectData를 돌려준다.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.dto import ObservationDTO
from agent.prompts import LOCATE_PROMPT, MENTIONS_PROMPT
from agent.structured import structured_call
from agent.unity_client import UnityClient
from agent.world_text import object_kind

# 시야각(70도)보다 촘촘하게 돌아보는 고개 방향(도). 정면부터 본다.
SCAN_YAWS = (0.0, 60.0, -60.0, 120.0, -120.0, 180.0)
# 객체가 아닌 말. 이미지에서 찾지 않는다.
NOT_OBJECTS = {"you", "robot", "agent", "me", "ground", "floor", "here", "there", "hand", "inventory", "it", "them"}
EMPTY_NAMES = {"", "null", "none", "nothing", "n/a"}


class Mention(BaseModel):
    name: str = Field(description="Object noun from the command, e.g. barrel")
    label: str | None = Field(default=None, description="Unused; labels are matched by code")


class Mentions(BaseModel):
    objects: list[Mention] = Field(default_factory=list, description="Object nouns that appear in the command")


class Sighting(BaseModel):
    visible: bool = Field(description="Whether the object is visible in this image")
    x: int = Field(default=0, ge=0, le=1000, description="Center x, 0 = left edge, 1000 = right edge")
    y: int = Field(default=0, ge=0, le=1000, description="Center y, 0 = top edge, 1000 = bottom edge")


@dataclass
class GroundingItem:
    name: str
    status: str  # known | found | not_found
    label: str | None = None
    yaw: float | None = None
    point: tuple[int, int] | None = None
    feedback: str = ""

    def note(self) -> str:
        if self.status == "not_found":
            return f"- '{self.name}' is not in the object list and was not found by looking around; it may not exist here."
        return f"- '{self.name}' = {self.label}"


@dataclass
class GroundingReport:
    items: list[GroundingItem]

    @property
    def registered(self) -> list[GroundingItem]:
        return [item for item in self.items if item.status == "found"]

    def notes(self) -> str:
        lines = [item.note() for item in self.items if item.status != "known"]
        return ("[grounding] objects found by looking around:\n" + "\n".join(lines)) if lines else ""

    def as_record(self) -> list[dict]:
        return [asdict(item) for item in self.items]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def classify(mentions: list[Mention], observation: ObservationDTO) -> tuple[list[GroundingItem], list[str]]:
    """known 항목과, 이미지에서 찾아야 할 이름 목록을 돌려준다."""
    labels = {o.label for o in observation.objects}
    kinds = {object_kind(label) for label in labels}
    known: list[GroundingItem] = []
    unknown: list[str] = []
    for mention in mentions:
        name = mention.name.strip().lower()
        if name in labels:
            known.append(GroundingItem(name=name, status="known", label=name))
            continue
        words = _words(name)
        if not words or name in NOT_OBJECTS or name in EMPTY_NAMES:
            continue
        singular = {w[:-1] if w.endswith("s") else w for w in words}
        # 모델이 붙인 label은 믿지 않는다. 이름에 아는 종류가 들어 있으면 known, 아니면 이미지에서 찾는다.
        if kinds & (set(words) | singular):
            known.append(GroundingItem(name=name, status="known"))
        elif name not in unknown:
            unknown.append(name)
    return known, unknown


async def find_in_view(llm: BaseChatModel, image_png_base64: str, name: str) -> Sighting:
    return await structured_call(
        llm,
        Sighting,
        [
            SystemMessage(LOCATE_PROMPT),
            HumanMessage(
                content=[
                    {"type": "text", "text": f"Is there a {name} in this image? If yes, give the point at its center."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_png_base64}"}},
                ]
            ),
        ],
    )


async def locate_unknown(llm: BaseChatModel, unity: UnityClient, name: str, first_view: str) -> GroundingItem:
    for yaw in SCAN_YAWS:
        image = first_view if yaw == 0.0 else await unity.view(yaw)
        sighting = await find_in_view(llm, image, name)
        if not sighting.visible:
            continue
        result = await unity.locate(yaw, sighting.x / 1000, sighting.y / 1000, name)
        if result.found and result.object is not None:
            return GroundingItem(
                name=name,
                status="found",
                label=result.object.label,
                yaw=yaw,
                point=(sighting.x, sighting.y),
                feedback=result.feedback,
            )
    return GroundingItem(name=name, status="not_found", feedback="not visible in any view")


async def ground_objects(
    llm: BaseChatModel, unity: UnityClient, command: str, observation: ObservationDTO
) -> GroundingReport:
    labels = ", ".join(o.label for o in observation.objects)
    mentions = await structured_call(
        llm,
        Mentions,
        [SystemMessage(MENTIONS_PROMPT), HumanMessage(content=f"Object list: {labels}\nCommand: {command}")],
    )
    items, unknown = classify(mentions.objects, observation)
    for name in unknown:
        items.append(await locate_unknown(llm, unity, name, observation.image_png_base64))
    return GroundingReport(items)
