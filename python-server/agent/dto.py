"""Unity 행동 서버와 주고받는 데이터. 필드 이름은 Unity AgentDtos.cs와 같다."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Dto(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Vec3(_Dto):
    x: float
    y: float
    z: float

    def horizontal_distance(self, other: "Vec3") -> float:
        return ((self.x - other.x) ** 2 + (self.z - other.z) ** 2) ** 0.5

    def __str__(self) -> str:
        return f"({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"


class ObjectDataDTO(_Dto):
    """Unity ObjectData와 1:1. label로 객체를 특정하고, pickable은 집어 옮길 수 있는지(속성)다."""

    label: str
    position: Vec3
    pickable: bool = True


class AgentStateDTO(_Dto):
    position: Vec3
    holding: str | None = None
    inventory: list[str] = Field(default_factory=list)


class ObservationDTO(_Dto):
    timestamp: float
    agent: AgentStateDTO
    objects: list[ObjectDataDTO]
    message: str | None = None
    image_png_base64: str

    def find(self, label: str) -> ObjectDataDTO | None:
        return next((o for o in self.objects if o.label == label), None)


ActionName = Literal["move_to", "look_at", "pick", "store", "drop"]


class ActRequestDTO(_Dto):
    action: ActionName
    label: str | None = None
    near_label: str | None = None


class ActionResultDTO(_Dto):
    action: str
    success: bool
    feedback: str
    started_at: float
    finished_at: float
    observation: ObservationDTO


class LocateResultDTO(_Dto):
    """Unity /perceive/locate 응답. 새로 등록했거나(known=false) 이미 알던(known=true) 객체의 ObjectData."""

    found: bool
    known: bool = False
    feedback: str = ""
    object: ObjectDataDTO | None = None
