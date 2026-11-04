"""Thread API(FastAPI). Unity 채팅 UI와 HTTP 클라이언트가 같은 API를 쓴다."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.config import Settings
from agent.service import AgentService


class CommandIn(BaseModel):
    text: str = Field(min_length=1)


def create_app(settings: Settings, service: AgentService | None = None) -> FastAPI:
    service = service or AgentService(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await service.start()
        yield
        await service.stop()

    app = FastAPI(title="Carrier Agent cognition server", lifespan=lifespan)
    app.state.service = service

    @app.get("/health")
    async def health() -> dict:
        try:
            unity = await service.unity.health()
        except Exception as exc:  # noqa: BLE001
            unity = {"status": "unreachable", "error": str(exc)}
        return {
            "status": "ok",
            "model_profile": settings.profile.name,
            "model": settings.profile.model,
            "judge_profile": settings.judge_profile.name,
            "unity": unity,
        }

    @app.post("/threads")
    async def create_thread() -> dict:
        return {"thread_id": service.create_thread().thread_id}

    @app.post("/threads/{thread_id}/commands")
    async def submit_command(thread_id: str, command: CommandIn) -> dict:
        try:
            record = service.submit(thread_id, command.text)
        except KeyError:
            raise HTTPException(404, f"Unknown thread {thread_id}") from None
        return {"command_id": record.command_id, "status": record.status}

    @app.get("/threads/{thread_id}")
    async def get_thread(thread_id: str) -> dict:
        thread = service.threads.get(thread_id)
        if thread is None:
            raise HTTPException(404, f"Unknown thread {thread_id}")
        return thread.view()

    @app.post("/commands/{command_id}/cancel")
    async def cancel_command(command_id: str) -> dict:
        try:
            return {"cancelled": await service.cancel(command_id)}
        except KeyError:
            raise HTTPException(404, f"Unknown command {command_id}") from None

    return app
