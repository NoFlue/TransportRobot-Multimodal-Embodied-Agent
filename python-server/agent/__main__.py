"""CLI: python -m agent serve | chat | run "명령" | reset | experiment"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agent.config import load_settings
from agent.service import AgentService
from agent.unity_client import UnityClient


def _print_event(_: str, message: dict) -> None:
    if message["role"] != "user":
        print(f"[{message['role']}] {message['text']}", flush=True)


async def _chat(service: AgentService) -> None:
    service.listeners.append(_print_event)
    await service.start()
    thread = service.create_thread()
    print("Type a command. /new starts a new conversation, /reset resets the Unity world, /quit exits.")
    try:
        while True:
            text = (await asyncio.to_thread(input, "you> ")).strip()
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text == "/new":
                thread = service.create_thread()
                print("New conversation.")
                continue
            if text == "/reset":
                await service.unity.reset_world()
                print("Unity world reset.")
                continue
            record = await service.run(thread.thread_id, text)
            print(f"-- {record.status}, steps={record.steps}, judge={record.judge_success}")
    finally:
        await service.stop()


async def _run(service: AgentService, text: str, reset: bool) -> int:
    service.listeners.append(_print_event)
    await service.start()
    try:
        if reset:
            await service.unity.reset_world()
        thread = service.create_thread()
        record = await service.run(thread.thread_id, text)
        print(json.dumps(record.view(), ensure_ascii=False, indent=2))
        print(f"run folder: {service.recorder.run_dir}")
        return 0 if record.status == "done" else 1
    finally:
        await service.stop()


async def _reset(settings) -> None:
    unity = UnityClient(settings.app.unity)
    try:
        await unity.reset_world()
        print("Unity world reset.")
    finally:
        await unity.aclose()


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="python -m agent")
    parser.add_argument("--profile", help="model profile in config/models.yaml")
    parser.add_argument("--judge-profile", help="judge model profile (default: same as --profile)")
    parser.add_argument("--planner-profile", help="planner model profile, or 'none' to act without a plan")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="run the HTTP thread API")
    commands.add_parser("chat", help="interactive conversation in the terminal")
    run = commands.add_parser("run", help="run one command and exit")
    run.add_argument("text")
    run.add_argument("--reset", action="store_true", help="reset the Unity world first")
    commands.add_parser("reset", help="reset the Unity world")
    experiment = commands.add_parser("experiment", help="run the success-rate experiment in config/experiment.yaml")
    experiment.add_argument("--suite", type=Path, help="test definition file (default: config/experiment.yaml)")
    experiment.add_argument("--areas", help="comma-separated area ids to run, e.g. double,complex")
    experiment.add_argument("--trials", type=int, help="trials per area (default: value in the suite)")
    experiment.add_argument("--out", type=Path, help="output folder; reuse one to keep appending to its summary.csv")
    experiment.add_argument("--replay", type=Path, help="reuse seeds, categories and commands from another experiment folder")
    args = parser.parse_args(argv)

    settings = load_settings(profile=args.profile, judge_profile=args.judge_profile, planner_profile=args.planner_profile)
    if args.command == "serve":
        import uvicorn

        from agent.server import create_app

        uvicorn.run(create_app(settings), host=settings.app.server.host, port=settings.app.server.port)
        return 0
    if args.command == "reset":
        asyncio.run(_reset(settings))
        return 0
    if args.command == "experiment":
        from agent.experiment import run_experiment

        suite = args.suite or settings.config_dir / "experiment.yaml"
        areas = [area.strip() for area in args.areas.split(",")] if args.areas else None
        asyncio.run(run_experiment(settings, suite, areas=areas, trials=args.trials, out_dir=args.out, replay=args.replay))
        return 0
    service = AgentService(settings)
    if args.command == "chat":
        asyncio.run(_chat(service))
        return 0
    return asyncio.run(_run(service, args.text, args.reset))


if __name__ == "__main__":
    raise SystemExit(main())
