"""논문 실험 러너: python -m agent experiment

config/experiment.yaml의 영역(단일·이중·복합·추상 명령)마다 명령을 trials회 실행하고 측정값을 CSV로 남긴다.
- 성공률: 성공 시행 수 / 전체 시행 수
- 평균 완료 시간: 명령 수신 → 에이전트가 명령 처리를 끝낸 시점. 성공한 시행만 평균에 넣는다.
- 단계 완료율: 정상 완료한 필수 행동 단계 수 합 / 전체 필수 행동 단계 수 합 (measure_steps=false 영역은 비운다)
성공은 규칙으로 정한다: 정상 종료(answered) + 필수 단계 전부 완료 + 최종 상태 조건(goals) 충족.
요청보다 많은 물체를 옮겨도(집는 단계 수 + 명령이 내려놓으라고 한 준비 물체 수를 넘으면) 실패다.
시행마다 Unity 월드를 리셋하고 새 대화(thread)에서 시작한다. generator가 있는 영역은 시행마다 현재 시각(ms)을 시드로
객체 배치를 바꾸고, 같은 시드로 agent/command_gen.py가 명령을 만든다. 시드는 시행 CSV에 남는다.
"""

from __future__ import annotations

import asyncio
import csv
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, Field

from agent import command_gen
from agent.config import Settings
from agent.dto import ActRequestDTO, ObservationDTO
from agent.graph import StepRecord
from agent.metrics import RunRecorder
from agent.service import AgentService
from agent.world_text import object_kind

ANY = "any"
GOAL_KINDS = {"agent_near", "near", "carrying", "inventory", "empty_handed", "not_carrying"}
YES = {"y", "yes", "예", "네", "응", "ㅇ"}

TRIAL_FIELDS = [
    "area",
    "trial",
    "seed",
    "category",
    "command",
    "success",
    "elapsed_seconds",
    "completed_steps",
    "required_steps",
    "step_completion_percent",
    "end_reason",
    "unmet_goals",
    "moved_objects",
    "allowed_moves",
    "actions",
    "reference",
    "reply",
    "command_id",
]
SUMMARY_FIELDS = [
    "area",
    "name",
    "trials",
    "successes",
    "success_rate_percent",
    "avg_completion_seconds",
    "std_completion_seconds",
    "completed_steps",
    "required_steps",
    "step_completion_percent",
    "finished_at",
]
CATEGORY_FIELDS = ["area", "category", "trials", "successes", "success_rate_percent", "avg_completion_seconds"]


class StepSpec(BaseModel):
    tool: str
    label: str | None = None
    near_label: str | None = None


class TaskSpec(BaseModel):
    command: str
    steps: list[StepSpec]
    ordered: bool = True
    goals: list[dict[str, Any]] = Field(default_factory=list)
    setup: list[ActRequestDTO] = Field(default_factory=list)
    reference: list[ActRequestDTO] = Field(default_factory=list)  # 정답 행동(생성기가 만든 경우)


class AreaSpec(BaseModel):
    id: str
    name: str
    measure_steps: bool = True
    tasks: list[TaskSpec] = Field(default_factory=list)
    generator: str | None = None  # single | double | complex | abstract: 시행마다 명령을 만든다
    quota: dict[str, int] = Field(default_factory=dict)  # single 생성기의 유형별 시행 수


class Suite(BaseModel):
    trials: int = 20
    seeded: bool = False  # 고정 명령 영역도 시행마다 시드로 배치를 바꿀지
    areas: list[AreaSpec]


def load_suite(path: Path) -> Suite:
    with path.open(encoding="utf-8") as stream:
        suite = Suite.model_validate(yaml.safe_load(stream))
    for area in suite.areas:
        if not area.tasks and not area.generator:
            raise ValueError(f"Area '{area.id}' needs tasks or a generator")
        for task in area.tasks:
            for goal in task.goals:
                if len(goal) != 1 or next(iter(goal)) not in GOAL_KINDS:
                    raise ValueError(f"Bad goal {goal} in '{task.command}'. Use one of: {', '.join(sorted(GOAL_KINDS))}")
    return suite


# ---------------- 판정 ----------------


def label_matches(pattern: str | None, label: str | None) -> bool:
    """pattern: 정확한 label(coal_1), 종류(coal), any, 또는 None(검사하지 않음)."""
    if pattern is None:
        return True
    if not label:
        return False
    return pattern == ANY or label == pattern or object_kind(label) == pattern


def step_matches(spec: StepSpec, record: StepRecord) -> bool:
    return (
        record.tool_success
        and record.tool_name == spec.tool
        and label_matches(spec.label, record.tool_args.get("label"))
        and label_matches(spec.near_label, record.tool_args.get("near_label"))
    )


def completed_steps(task: TaskSpec, records: list[StepRecord]) -> int:
    """필수 단계 중 Unity가 성공으로 보고한 호출과 맞춘 단계 수."""
    if not task.ordered:
        used: set[int] = set()
        for spec in task.steps:
            index = next((i for i, r in enumerate(records) if i not in used and step_matches(spec, r)), None)
            if index is not None:
                used.add(index)
        return len(used)
    # 순서가 있는 명령: 필수 단계 목록과 실제 호출 목록의 최장 공통 부분열 길이
    table = [[0] * (len(records) + 1) for _ in range(len(task.steps) + 1)]
    for i, spec in enumerate(task.steps, 1):
        for j, record in enumerate(records, 1):
            if step_matches(spec, record):
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def unmet_goals(goals: list[dict[str, Any]], observation: ObservationDTO, radius: float) -> list[str]:
    agent = observation.agent
    carried = [label for label in [agent.holding, *agent.inventory] if label]
    ground = [o for o in observation.objects if o.label not in carried]
    unmet = []
    for goal in goals:
        (kind, value), = goal.items()
        if kind == "agent_near":
            ok = any(label_matches(value, o.label) and agent.position.horizontal_distance(o.position) <= radius for o in ground)
        elif kind == "near":
            item, target = value
            ok = any(
                a.label != b.label
                and a.pickable
                and label_matches(item, a.label)
                and label_matches(target, b.label)
                and a.position.horizontal_distance(b.position) <= radius
                for a in ground
                for b in ground
            )
        elif kind == "carrying":
            ok = any(label_matches(value, label) for label in carried)
        elif kind == "inventory":
            ok = any(label_matches(value, label) for label in agent.inventory)
        elif kind == "not_carrying":
            ok = not any(label_matches(value, label) for label in carried)
        else:  # empty_handed
            ok = not carried
        if not ok:
            unmet.append(f"{kind}={value}")
    return unmet


def count_moved(before: ObservationDTO, after: ObservationDTO, threshold: float = 0.5) -> int:
    """명령 동안 상태가 바뀐 집을 수 있는 객체 수. 들고 있음 여부가 바뀌거나 땅 위 위치가 threshold 이상 바뀌면 센다."""
    def carried(obs: ObservationDTO) -> set[str]:
        return {label for label in [obs.agent.holding, *obs.agent.inventory] if label}

    held_before, held_after = carried(before), carried(after)
    final = {o.label: o for o in after.objects}
    moved = 0
    for obj in before.objects:
        if not obj.pickable or obj.label not in final:
            continue
        was, now = obj.label in held_before, obj.label in held_after
        if was != now or (not was and obj.position.horizontal_distance(final[obj.label].position) > threshold):
            moved += 1
    return moved


def allowed_moves(task: TaskSpec, setup: list[ActRequestDTO]) -> int:
    """명령이 상태를 바꿔도 되는 물체 수: 집는 단계 수 + 준비로 들고 있던 물체 중 내려놓으라고 한 수."""
    held = {action.label for action in setup if action.action == "pick"}
    picks = sum(1 for step in task.steps if step.tool == "pick")
    drops = sum(1 for step in task.steps if step.tool == "drop" and step.label in held)
    return picks + drops


# ---------------- 실행 ----------------


class _StepCollector(RunRecorder):
    """기존 기록(steps.csv, results.csv, 이미지)을 그대로 남기면서 판정용으로 명령별 단계를 모은다."""

    def __init__(self, run_dir: Path, save_images: bool):
        super().__init__(run_dir, save_images)
        self.run_dir = run_dir
        self.steps: dict[str, list[StepRecord]] = {}

    def record_step(self, thread_id: str, command_id: str, record: StepRecord) -> None:
        self.steps.setdefault(command_id, []).append(record)
        super().record_step(thread_id, command_id, record)


def _append_csv(path: Path, fields: list[str], row: dict) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig" if is_new else "utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _format_calls(calls: list[ActRequestDTO]) -> str:
    return " -> ".join(f"{c.action}({', '.join(v for v in (c.label, c.near_label) if v)})" for c in calls)


async def _run_trial(
    service: AgentService,
    recorder: _StepCollector,
    area: AreaSpec,
    task: TaskSpec | None,
    trial: int,
    seed: int | None,
    category: str,
) -> dict:
    unity = service.unity
    await unity.reset_world(seed)
    draft = None
    if area.generator:
        draft = command_gen.draft(area.generator, category, random.Random(seed), await unity.observe())
        setup = [ActRequestDTO.model_validate(action) for action in draft.setup]
    else:
        setup = task.setup
    for action in setup:
        result = await unity.act(action)
        if not result.success:
            raise RuntimeError(f"Setup {action.action}({action.label}) failed: {result.feedback}")
    before = await unity.observe()
    if draft is not None:
        spec = draft.finalize(before)
        category = spec.pop("category", category)
        task = TaskSpec.model_validate(spec)
    thread = service.create_thread()
    record = await service.run(thread.thread_id, task.command)
    steps = recorder.steps.pop(record.command_id, [])
    observation = await unity.observe()
    done = completed_steps(task, steps)
    required = len(task.steps)
    unmet = unmet_goals(task.goals, observation, service.settings.app.agent.near_radius_meters)
    moved = count_moved(before, observation)
    allowed = allowed_moves(task, setup)
    success = record.end_reason == "answered" and done == required and not unmet and moved <= allowed
    actions = [f"{s.tool_name}({', '.join(f'{v}' for v in s.tool_args.values() if v)}):{'ok' if s.tool_success else 'fail'}" for s in steps]
    return {
        "area": area.id,
        "trial": trial,
        "seed": "" if seed is None else seed,
        "category": category,
        "command": task.command,
        "success": success,
        "elapsed_seconds": round(record.total_seconds or 0.0, 3),
        "completed_steps": done,
        "required_steps": required,
        "step_completion_percent": round(100 * done / required, 1),
        "end_reason": record.end_reason,
        "unmet_goals": "; ".join(unmet),
        "moved_objects": moved,
        "allowed_moves": allowed,
        "actions": " -> ".join(actions),
        "reference": _format_calls(task.reference),
        "reply": record.reply,
        "command_id": record.command_id,
    }


def summarize(area: AreaSpec, rows: list[dict]) -> dict:
    times = [row["elapsed_seconds"] for row in rows if row["success"]]
    done = sum(row["completed_steps"] for row in rows)
    required = sum(row["required_steps"] for row in rows)
    steps = area.measure_steps and required > 0
    return {
        "area": area.id,
        "name": area.name,
        "trials": len(rows),
        "successes": len(times),
        "success_rate_percent": round(100 * len(times) / len(rows), 1) if rows else "",
        "avg_completion_seconds": round(statistics.mean(times), 3) if times else "",
        "std_completion_seconds": round(statistics.stdev(times), 3) if len(times) > 1 else "",
        "completed_steps": done if steps else "",
        "required_steps": required if steps else "",
        "step_completion_percent": round(100 * done / required, 1) if steps else "",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def _print_trial(row: dict, trials: int) -> None:
    outcome = "성공" if row["success"] else "실패"
    seed = f" seed={row['seed']}" if row["seed"] != "" else ""
    line = (
        f"[{row['area']} {row['trial']:02d}/{trials}] {outcome}  {row['elapsed_seconds']:.1f}s  "
        f"steps {row['completed_steps']}/{row['required_steps']}{seed} [{row['category']}]  {row['command']}"
    )
    if not row["success"]:
        line += (
            f"\n    end_reason={row['end_reason']} unmet=[{row['unmet_goals']}] moved={row['moved_objects']}/{row['allowed_moves']}"
            f" actions={row['actions'] or '-'}"
        )
    print(line, flush=True)


class _SeedClock:
    """현재 시각(ms)으로 시드를 만든다. 같은 ms에 두 번 부르면 1을 더해 겹치지 않게 한다."""

    def __init__(self) -> None:
        self.last = -1

    def next(self) -> int:
        seed = int(time.time() * 1000) % 2**31
        seed = seed if seed != self.last else seed + 1
        self.last = seed
        return seed


def _categories(area: AreaSpec, trials: int, rng: random.Random) -> list[str]:
    """quota가 있으면 유형별 개수만큼 만들어 섞는다. 시행 수가 다르면 앞에서부터 쓰거나 되풀이한다."""
    if not area.quota:
        return [area.generator or ""] * trials
    pool = [name for name, count in area.quota.items() for _ in range(count)]
    rng.shuffle(pool)
    return [pool[i % len(pool)] for i in range(trials)]


def summarize_categories(area: AreaSpec, rows: list[dict]) -> list[dict]:
    result = []
    for category in dict.fromkeys(row["category"] for row in rows):
        group = [row for row in rows if row["category"] == category]
        times = [row["elapsed_seconds"] for row in group if row["success"]]
        result.append({
            "area": area.id,
            "category": category,
            "trials": len(group),
            "successes": len(times),
            "success_rate_percent": round(100 * len(times) / len(group), 1),
            "avg_completion_seconds": round(statistics.mean(times), 3) if times else "",
        })
    return result


async def run_experiment(
    settings: Settings,
    suite_path: Path,
    *,
    areas: list[str] | None = None,
    trials: int | None = None,
    out_dir: Path | None = None,
    replay: Path | None = None,
    ask: Callable[[str], str] = input,
) -> Path:
    suite = load_suite(suite_path)
    unknown = set(areas or []) - {area.id for area in suite.areas}
    if unknown:
        raise ValueError(f"Unknown area(s) {', '.join(sorted(unknown))}. Choose from: {', '.join(a.id for a in suite.areas)}")
    selected = [area for area in suite.areas if not areas or area.id in areas]
    trials = trials or suite.trials
    seeds = _SeedClock()
    out_dir = out_dir or settings.config_dir.parent / "experiments" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 성공은 규칙으로 판정한다. LLM 판정을 끄면 total_seconds가 명령 수신 → 에이전트 종료 시간이 된다.
    settings.app.judge.enabled = False
    recorder = _StepCollector(out_dir / "logs", settings.app.logging.save_images)
    service = AgentService(settings, recorder=recorder)
    await service.start()
    planner = settings.planner_profile.name if settings.planner_profile else "none (pure ReAct)"
    print(f"experiment folder: {out_dir}\nmodel: {settings.profile.model}  planner: {planner}  trials per area: {trials}", flush=True)
    if replay:
        print(f"replay: {replay} 의 시드·유형·명령을 그대로 다시 쓴다", flush=True)
    try:
        for index, area in enumerate(selected):
            print(f"\n=== {area.name} ({area.id}) {trials}회 ===", flush=True)
            seeded = bool(area.generator) or suite.seeded
            recorded = _replay_rows(replay, area, trials) if replay else None
            if recorded is not None:
                categories = [row["category"] for row in recorded]
            else:
                order_seed = seeds.next()
                categories = _categories(area, trials, random.Random(order_seed))
                if area.quota:
                    print(f"유형 순서 시드 {order_seed}: " + ", ".join(f"{k} {categories.count(k)}" for k in area.quota), flush=True)
            rows = []
            mismatches = 0
            for trial in range(1, trials + 1):
                task = area.tasks[(trial - 1) % len(area.tasks)] if area.tasks else None
                if recorded is not None:
                    seed = int(recorded[trial - 1]["seed"]) if recorded[trial - 1]["seed"] else None
                else:
                    seed = seeds.next() if seeded else None
                row = await _run_trial(service, recorder, area, task, trial, seed, categories[trial - 1])
                if recorded is not None and row["command"] != recorded[trial - 1]["command"]:
                    mismatches += 1
                    print(f"    [replay] 명령이 원래 기록과 다름: {recorded[trial - 1]['command']!r}", flush=True)
                rows.append(row)
                _append_csv(out_dir / f"{area.id}_trials.csv", TRIAL_FIELDS, row)
                _print_trial(row, trials)
            if recorded is not None:
                print(f"replay 명령 불일치 {mismatches}/{trials}", flush=True)
            summary = summarize(area, rows)
            _append_csv(out_dir / "summary.csv", SUMMARY_FIELDS, summary)
            for category_row in summarize_categories(area, rows):
                _append_csv(out_dir / "category_summary.csv", CATEGORY_FIELDS, category_row)
            step_text = f", 단계 완료율 {summary['step_completion_percent']}%" if area.measure_steps else ""
            print(
                f"--- {area.name}: 성공 {summary['successes']}/{summary['trials']} ({summary['success_rate_percent']}%), "
                f"평균 완료 시간 {summary['avg_completion_seconds'] or '-'}s{step_text}",
                flush=True,
            )
            if index + 1 < len(selected):
                following = selected[index + 1]
                answer = await asyncio.to_thread(ask, f"다음 테스트 [{following.name}]를 진행할까요? (y/n) ")
                if answer.strip().lower() not in YES:
                    rest = ",".join(a.id for a in selected[index + 1 :])
                    print(f"중단했습니다. 이어서 하려면: python -m agent experiment --areas {rest} --out \"{out_dir}\"")
                    break
    finally:
        await service.stop()
    return out_dir


def _replay_rows(folder: Path, area: AreaSpec, trials: int) -> list[dict]:
    """다른 실험 폴더의 시행 기록(시드·유형·명령)을 순서대로 읽는다. 같은 배치와 명령으로 짝지어 비교할 때 쓴다."""
    with (folder / f"{area.id}_trials.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < trials:
        raise ValueError(f"{folder / f'{area.id}_trials.csv'} has {len(rows)} trials, fewer than {trials}")
    return rows[:trials]

