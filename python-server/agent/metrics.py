"""단계 기록(steps.csv)과 명령 기록(results.csv)을 실행 폴더에 남긴다."""

from __future__ import annotations

import base64
import csv
import json
from datetime import datetime
from pathlib import Path

from agent.graph import StepRecord

STEP_FIELDS = [
    "thread_id",
    "command_id",
    "step",
    "tool_name",
    "tool_args",
    "reasoning_seconds",
    "step_to_call_seconds",
    "execution_seconds",
    "tool_success",
    "feedback",
]
RESULT_FIELDS = [
    "thread_id",
    "command_id",
    "text",
    "model_profile",
    "judge_profile",
    "planner_profile",
    "planning_seconds",
    "command_to_first_action_seconds",
    "judge_success",
    "judge_rationale",
    "steps",
    "tool_success_rate",
    "end_reason",
    "total_seconds",
]


def _round(value: float | None) -> float | str:
    return "" if value is None else round(value, 4)


class RunRecorder:
    def __init__(self, runs_dir: Path, save_images: bool):
        self.run_dir = runs_dir / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.save_images = save_images

    def _append(self, name: str, fields: list[str], row: dict) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / name
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8-sig" if is_new else "utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    def record_step(self, thread_id: str, command_id: str, record: StepRecord) -> None:
        self._append(
            "steps.csv",
            STEP_FIELDS,
            {
                "thread_id": thread_id,
                "command_id": command_id,
                "step": record.step,
                "tool_name": record.tool_name,
                "tool_args": json.dumps(record.tool_args, ensure_ascii=False),
                "reasoning_seconds": _round(record.reasoning_seconds),
                "step_to_call_seconds": _round(record.step_to_call_seconds),
                "execution_seconds": _round(record.execution_seconds),
                "tool_success": record.tool_success,
                "feedback": record.feedback,
            },
        )
        if self.save_images and record.image_png_base64:
            image_dir = self.run_dir / "images" / command_id
            image_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / f"step_{record.step:03d}.png").write_bytes(base64.b64decode(record.image_png_base64))

    def record_result(self, row: dict) -> None:
        self._append("results.csv", RESULT_FIELDS, {key: row.get(key, "") for key in RESULT_FIELDS})

    def record_plan(self, thread_id: str, command_id: str, plan: dict) -> None:
        """계획 시도마다 한 줄: 다시 쓴 명령, 계획, 검사 오류, 모델 원문. 행동이 이상할 때 원인을 찾는 데 쓴다."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"thread_id": thread_id, "command_id": command_id, **plan}, ensure_ascii=False)
        with (self.run_dir / "plans.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
