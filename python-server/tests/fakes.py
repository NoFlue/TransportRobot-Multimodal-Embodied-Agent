"""테스트용 가짜 Unity 세계와 대본대로 답하는 모델."""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


class FakeUnityWorld:
    """CarrierAgentRuntime의 HTTP 동작을 단순하게 흉내 낸다."""

    def __init__(self) -> None:
        self.clock = 100.0
        self.agent = [0.0, -22.0]
        self.holding: str | None = None
        self.inventory: list[str] = []
        self.objects = {
            "coal_1": [-17.0, 14.0],
            "furnace_1": [0.0, 4.0],
            "iron_1": [17.0, 15.0],
        }
        self.act_calls: list[dict] = []
        self.blocked: set[tuple[str, str]] = set()  # 실패시킬 (action, label)
        self.fixed: set[str] = {"furnace_1"}
        self.hidden: dict[str, tuple[float, list[float]]] = {}  # 이름 -> (보이는 yaw, 위치). 목록에 없는 객체
        self.view_calls: list[float] = []
        self.locate_calls: list[dict] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.clock += 0.01
        route = f"{request.method} {request.url.path}"
        if route == "GET /clock":
            return httpx.Response(200, json={"unity_time": self.clock})
        if route == "GET /observe":
            return httpx.Response(200, json=self.observation())
        if route == "POST /act":
            return httpx.Response(200, json=self.act(json.loads(request.content)))
        if route == "POST /perceive/view":
            yaw = json.loads(request.content)["yaw"]
            self.view_calls.append(yaw)
            return httpx.Response(200, json={"yaw": yaw, "image_png_base64": PNG})
        if route == "POST /perceive/locate":
            return httpx.Response(200, json=self.locate(json.loads(request.content)))
        if route in ("POST /act/cancel", "POST /world/reset"):
            return httpx.Response(200, json={"cancelled": False, "status": "reset"})
        if route == "GET /health":
            return httpx.Response(200, json={"status": "ok", "busy": False})
        return httpx.Response(404, json={"error": route})

    def observation(self) -> dict[str, Any]:
        def pos(xz):
            return {"x": xz[0], "y": 0.0, "z": xz[1]}

        objects = []
        for label, xz in sorted(self.objects.items()):
            carried = label == self.holding or label in self.inventory
            objects.append({"label": label, "position": pos(self.agent if carried else xz), "pickable": label not in self.fixed})
        return {
            "timestamp": self.clock,
            "agent": {"position": pos(self.agent), "holding": self.holding, "inventory": list(self.inventory)},
            "objects": objects,
            "message": None,
            "image_png_base64": PNG,
        }

    def locate(self, body: dict) -> dict:
        self.locate_calls.append(body)
        entry = self.hidden.get(body["name"])
        if entry is None or entry[0] != body["yaw"]:
            return {"found": False, "known": False, "feedback": "Only the ground is at that point of the view."}
        label = f"{body['name']}_1"
        self.objects[label] = entry[1]
        self.fixed.add(label)
        del self.hidden[body["name"]]
        position = {"x": entry[1][0], "y": 0.0, "z": entry[1][1]}
        return {"found": True, "known": False, "feedback": f"Registered as {label}.",
                "object": {"label": label, "position": position, "pickable": False}}

    def distance(self, label: str) -> float:
        x, z = self.objects[label]
        return math.dist((x, z), self.agent)

    def act(self, body: dict) -> dict:
        self.act_calls.append(body)
        started = self.clock
        action, label = body["action"], body.get("label")
        ok, feedback = True, "ok"
        if (action, label) in self.blocked:
            ok, feedback = False, "blocked"
        elif action == "move_to":
            x, z = self.objects[label]
            self.agent = [x, z - 1.0]
        elif action == "pick":
            # Unity처럼 멀면 걸어가고, 손이 차 있으면 들고 있던 객체를 인벤토리로 옮긴다.
            x, z = self.objects[label]
            self.agent = [x, z - 1.0]
            if self.holding is not None:
                self.inventory.append(self.holding)
            self.holding = label
        elif action == "store":
            self.inventory.append(self.holding)
            self.holding = None
        elif action == "drop":
            near = body.get("near_label")
            if label != self.holding and label not in self.inventory:
                ok, feedback = False, f"You are not carrying {label}."
            elif self.holding == label:
                self.holding = None
            else:
                self.inventory.remove(label)
            if ok:
                nx, nz = self.objects[near] if near else self.agent
                self.agent = [nx, nz - 1.0]
                self.objects[label] = [nx + 1.5, nz]
        self.clock += 2.0
        return {
            "action": action,
            "success": ok,
            "feedback": feedback,
            "started_at": started,
            "finished_at": self.clock,
            "observation": self.observation(),
        }


class ScriptedChatModel(BaseChatModel):
    """미리 정한 응답을 순서대로 돌려주고, 받은 프롬프트를 기록한다."""

    replies: list[AIMessage]
    prompts: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001 - 테스트 대역
        return self

    def with_structured_output(self, schema, **kwargs):  # noqa: ANN001 - 텍스트 응답 경로를 쓴다
        raise NotImplementedError

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ANN001
        self.prompts.append(list(messages))
        reply = self.replies.pop(0)
        return ChatResult(generations=[ChatGeneration(message=reply)])


def call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{len(args)}_{id(args)}"}])
