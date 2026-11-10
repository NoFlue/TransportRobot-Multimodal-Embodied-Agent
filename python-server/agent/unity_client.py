"""Unity 행동 서버(CarrierAgentRuntime) HTTP 클라이언트."""

from __future__ import annotations

import time

import httpx

from agent.config import UnityConfig
from agent.dto import ActionResultDTO, ActRequestDTO, LocateResultDTO, ObservationDTO


class UnityError(RuntimeError):
    pass


class UnityClient:
    def __init__(self, config: UnityConfig, transport: httpx.AsyncBaseTransport | None = None):
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.request_timeout_seconds,
            transport=transport,
        )
        # Unity 시계 - Python 시계(time.perf_counter). sync_clock()이 채운다.
        self.clock_offset: float | None = None
        self.clock_rtt: float | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, *, json: dict | None = None, timeout: float | None = None) -> dict:
        try:
            response = await self._client.request(
                method, path, json=json, timeout=timeout or self._config.request_timeout_seconds
            )
        except httpx.HTTPError as exc:
            raise UnityError(f"Unity {method} {path} failed: {exc!r}") from exc
        if response.status_code != 200:
            raise UnityError(f"Unity {method} {path} returned {response.status_code}: {response.text}")
        return response.json()

    async def health(self) -> dict:
        return await self._request("GET", "/health")

    async def sync_clock(self, samples: int = 5) -> float:
        """왕복 시간이 가장 짧은 표본으로 시계 차이를 추정한다. 오차는 왕복 시간의 절반 이하."""
        best: tuple[float, float] | None = None
        for _ in range(samples):
            sent = time.perf_counter()
            unity_time = (await self._request("GET", "/clock"))["unity_time"]
            received = time.perf_counter()
            rtt = received - sent
            if best is None or rtt < best[0]:
                best = (rtt, unity_time - (sent + received) / 2)
        assert best is not None
        self.clock_rtt, self.clock_offset = best
        return self.clock_offset

    def to_local_time(self, unity_time: float) -> float:
        if self.clock_offset is None:
            raise UnityError("Clock is not synchronized; call sync_clock() first")
        return unity_time - self.clock_offset

    async def observe(self) -> ObservationDTO:
        return ObservationDTO.model_validate(await self._request("GET", "/observe"))

    async def act(self, request: ActRequestDTO) -> ActionResultDTO:
        payload = request.model_dump(exclude_none=True)
        data = await self._request("POST", "/act", json=payload, timeout=self._config.action_timeout_seconds)
        return ActionResultDTO.model_validate(data)

    async def view(self, yaw: float) -> str:
        """몸은 그대로 두고 고개를 yaw도 돌린 시점 이미지(PNG base64)."""
        return (await self._request("POST", "/perceive/view", json={"yaw": yaw}))["image_png_base64"]

    async def locate(self, yaw: float, x: float, y: float, name: str) -> LocateResultDTO:
        """yaw 시점 이미지의 (x, y)(0~1, 왼쪽 위 기준)에 있는 객체를 특정해 등록한다."""
        payload = {"yaw": yaw, "x": x, "y": y, "name": name}
        return LocateResultDTO.model_validate(await self._request("POST", "/perceive/locate", json=payload))

    async def cancel(self) -> bool:
        return bool((await self._request("POST", "/act/cancel"))["cancelled"])

    async def reset_world(self, seed: int | None = None) -> None:
        """seed가 있으면 Unity가 그 시드로 객체 배치를 바꾼다. 없으면 처음 배치로 되돌린다."""
        await self._request("POST", "/world/reset", json=None if seed is None else {"seed": seed})
