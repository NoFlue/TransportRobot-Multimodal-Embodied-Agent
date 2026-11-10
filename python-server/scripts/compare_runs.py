"""두 실험(계획 구조 vs 순수 ReAct)을 같은 시드끼리 짝지어 비교하는 그림과 표.

사용: <python with matplotlib, numpy, scipy> scripts/compare_runs.py <계획 구조 실험 폴더> <순수 ReAct 실험 폴더> [출력 폴더]
두 번째 실험은 python -m agent --planner-profile none experiment ... --replay <첫 번째 폴더> 로 같은 시드·명령을 재생한 것이다.

통계
- 성공률: 95% Wilson 구간, 짝지은 비교는 McNemar 정확 검정(불일치 쌍에 대한 이항 검정, 양측)
- 완료 시간: 성공 시행의 평균 ± SD. 두 방식 모두 성공한 쌍에서 Wilcoxon 부호순위 검정
- 단계 완료율: 시행 단위 부트스트랩 95% 구간
- 도구 호출 수: 시행당 실제로 실행한 도구 호출 수, 불필요한 호출 = 실행 호출 수 - 필수 단계 수(0 미만은 0)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_figures import AREAS, TICKS, bootstrap_ratio, save, style, wilson  # noqa: E402

SYSTEMS = ["plan", "react"]
LABEL = {
    "ko": {"plan": "계획·검증 구조", "react": "순수 ReAct"},
    "en": {"plan": "Plan + validation", "react": "Pure ReAct"},
}
COLOR = {"plan": "#4C72B0", "react": "#DD8452"}
HATCH = {"plan": "", "react": "//"}
TEXT = {
    "ko": {"success": "성공률 (%)", "time": "완료 시간 (초, 성공 시행)", "steps": "단계 완료율 (%)", "calls": "시행당 도구 호출 수",
           "a": "(a) 성공률", "b": "(b) 완료 시간", "c": "(c) 단계 완료율"},
    "en": {"success": "Success rate (%)", "time": "Completion time (s), successful trials", "steps": "Step completion rate (%)",
           "calls": "Tool calls per trial", "a": "(a) Success rate", "b": "(b) Completion time", "c": "(c) Step completion"},
}
NAMES = {
    "ko": {"single": "단일 행동", "double": "이중 행동", "complex": "복합 행동", "abstract": "추상적 명령"},
    "en": {"single": "Single", "double": "Double", "complex": "Complex", "abstract": "Abstract"},
}


def read(folder: Path, area: str) -> list[dict] | None:
    path = folder / f"{area}_trials.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def failure_type(row: dict) -> str:
    if row["success"] == "True":
        return "success"
    if not row["actions"]:
        return "no_action"
    if int(row["moved_objects"]) > int(row["allowed_moves"]):
        return "over_transport"
    return "wrong_or_missing"


def calls(row: dict) -> int:
    return len([a for a in row["actions"].split(" -> ") if a]) if row["actions"] else 0


def area_stats(rows: list[dict]) -> dict:
    success = np.array([r["success"] == "True" for r in rows])
    times = np.array([float(r["elapsed_seconds"]) for r in rows])
    done = np.array([int(r["completed_steps"]) for r in rows])
    required = np.array([int(r["required_steps"]) for r in rows])
    n_calls = np.array([calls(r) for r in rows])
    k = int(success.sum())
    types = [failure_type(r) for r in rows]
    return {
        "n": len(rows), "k": k, "rate": k / len(rows), "rate_ci": wilson(k, len(rows)), "success": success,
        "times": times, "ok_times": times[success], "mean": float(times[success].mean()) if k else float("nan"),
        "sd": float(times[success].std(ddof=1)) if k > 1 else float("nan"),
        "done": int(done.sum()), "required": int(required.sum()), "step": done.sum() / required.sum(),
        "step_ci": bootstrap_ratio(done, required),
        "calls": float(n_calls.mean()), "extra": float(np.clip(n_calls - required, 0, None).mean()),
        "no_action": types.count("no_action"), "over_transport": types.count("over_transport"),
        "wrong_or_missing": types.count("wrong_or_missing"),
    }


def compare(plan_dir: Path, react_dir: Path) -> dict:
    result = {}
    for area in AREAS:
        a, b = read(plan_dir, area), read(react_dir, area)
        if not a or not b:
            continue
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        same = sum(x["seed"] == y["seed"] and x["command"] == y["command"] for x, y in zip(a, b))
        sa, sb = area_stats(a), area_stats(b)
        only_plan = int((sa["success"] & ~sb["success"]).sum())
        only_react = int((~sa["success"] & sb["success"]).sum())
        discordant = only_plan + only_react
        p = stats.binomtest(only_plan, discordant, 0.5).pvalue if discordant else 1.0
        both = sa["success"] & sb["success"]
        diff = sa["times"][both] - sb["times"][both]
        wilcoxon = stats.wilcoxon(diff).pvalue if both.sum() > 5 and np.any(diff) else float("nan")
        result[area] = {"plan": sa, "react": sb, "n": n, "matched": same, "only_plan": only_plan, "only_react": only_react,
                        "mcnemar_p": p, "both": int(both.sum()), "time_diff": float(diff.mean()) if both.any() else float("nan"),
                        "wilcoxon_p": wilcoxon}
    return result


def fmt_p(p: float) -> str:
    if p != p:
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def figure(data: dict, out: Path, lang: str) -> None:
    style(lang)
    areas = list(data)
    x = np.arange(len(areas))
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [4, 4, 3]})

    ax = axes[0]
    for i, system in enumerate(SYSTEMS):
        rates = [100 * data[a][system]["rate"] for a in areas]
        err = [[100 * (data[a][system]["rate"] - data[a][system]["rate_ci"][0]) for a in areas],
               [100 * (data[a][system]["rate_ci"][1] - data[a][system]["rate"]) for a in areas]]
        ax.bar(x + (i - 0.5) * width, rates, width, color=COLOR[system], hatch=HATCH[system], edgecolor="black", linewidth=0.5,
               yerr=err, capsize=2, error_kw={"elinewidth": 0.7}, label=LABEL[lang][system])
    for j, a in enumerate(areas):
        top = max(100 * data[a][s]["rate_ci"][1] for s in SYSTEMS)
        p = data[a]["mcnemar_p"]
        mark = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(j, top + 2, mark, ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel(TEXT[lang]["success"])

    ax = axes[1]
    for i, system in enumerate(SYSTEMS):
        values = [data[a][system]["ok_times"] for a in areas]
        box = ax.boxplot(values, positions=x + (i - 0.5) * width, widths=width * 0.85, patch_artist=True, showfliers=False,
                         medianprops={"color": "black", "linewidth": 0.9}, whiskerprops={"linewidth": 0.6},
                         capprops={"linewidth": 0.6}, boxprops={"linewidth": 0.6})
        for patch in box["boxes"]:
            patch.set_facecolor(COLOR[system])
            patch.set_alpha(0.55)
            patch.set_hatch(HATCH[system])
        ax.scatter(x + (i - 0.5) * width, [data[a][system]["mean"] for a in areas], marker="D", s=10, color="black", zorder=3)
    ax.set_ylim(0, None)
    ax.set_ylabel(TEXT[lang]["time"])

    ax = axes[2]
    step_areas = [a for a in areas if a != "single"]
    xs = np.arange(len(step_areas))
    for i, system in enumerate(SYSTEMS):
        values = [100 * data[a][system]["step"] for a in step_areas]
        err = [[100 * (data[a][system]["step"] - data[a][system]["step_ci"][0]) for a in step_areas],
               [100 * (data[a][system]["step_ci"][1] - data[a][system]["step"]) for a in step_areas]]
        ax.bar(xs + (i - 0.5) * width, values, width, color=COLOR[system], hatch=HATCH[system], edgecolor="black",
               linewidth=0.5, yerr=err, capsize=2, error_kw={"elinewidth": 0.7})
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel(TEXT[lang]["steps"])

    for ax, key, names in ((axes[0], "a", areas), (axes[1], "b", areas), (axes[2], "c", step_areas)):
        ax.set_title(TEXT[lang][key], loc="left")
        ax.set_xticks(np.arange(len(names)), [TICKS[lang][a] for a in names])
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.4)
    save(fig, out / f"fig_compare_{lang}")


def table(data: dict, out: Path, lang: str) -> None:
    if lang == "ko":
        header = ["영역", "방식", "성공", "성공률 % [95% CI]", "McNemar p", "완료 시간 평균 ± SD (초)", "단계 완료율 % [95% CI]",
                  "시행당 도구 호출", "불필요한 호출", "행동 없음", "과잉 운반", "단계 누락·대상 오류"]
    else:
        header = ["Category", "Method", "Successes", "Success % [95% CI]", "McNemar p", "Time mean ± SD (s)",
                  "Step completion % [95% CI]", "Tool calls/trial", "Extra calls/trial", "No action", "Over-transport",
                  "Missing/wrong step"]
    body = []
    for area, d in data.items():
        for system in SYSTEMS:
            s = d[system]
            step = "–" if area == "single" else f"{100 * s['step']:.1f} [{100 * s['step_ci'][0]:.1f}, {100 * s['step_ci'][1]:.1f}]"
            body.append([NAMES[lang][area] if system == "plan" else "", LABEL[lang][system], f"{s['k']}/{s['n']}",
                         f"{100 * s['rate']:.1f} [{100 * s['rate_ci'][0]:.1f}, {100 * s['rate_ci'][1]:.1f}]",
                         fmt_p(d["mcnemar_p"]) if system == "plan" else "", f"{s['mean']:.2f} ± {s['sd']:.2f}", step,
                         f"{s['calls']:.2f}", f"{s['extra']:.2f}", str(s["no_action"]), str(s["over_transport"]),
                         str(s["wrong_or_missing"])])
    with (out / f"table_compare_{lang}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        csv.writer(stream).writerows([header, *body])
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in body]
    (out / f"table_compare_{lang}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    esc = lambda s: s.replace("%", r"\%").replace("±", r"$\pm$").replace("–", "--").replace("<", r"$<$")  # noqa: E731
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\resizebox{\linewidth}{!}{%",
           r"\begin{tabular}{ll" + "r" * (len(header) - 2) + "}", r"\toprule", " & ".join(esc(h) for h in header) + r" \\",
           r"\midrule", *[" & ".join(esc(c) for c in r) + r" \\" for r in body], r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (out / f"table_compare_{lang}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def report(data: dict) -> str:
    lines = ["area | n | matched seed+command | plan success | react success | only plan | only react | McNemar p | "
             "both-success time diff (plan-react, s) | Wilcoxon p"]
    for area, d in data.items():
        lines.append(f"{area} | {d['n']} | {d['matched']} | {d['plan']['k']} | {d['react']['k']} | {d['only_plan']} | "
                     f"{d['only_react']} | {fmt_p(d['mcnemar_p'])} | {d['time_diff']:.2f} (n={d['both']}) | {fmt_p(d['wilcoxon_p'])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("react", type=Path)
    parser.add_argument("out", type=Path, nargs="?")
    args = parser.parse_args()
    data = compare(args.plan, args.react)
    out = args.out or args.react / "paper"
    out.mkdir(parents=True, exist_ok=True)
    for lang in ("ko", "en"):
        figure(data, out, lang)
        table(data, out, lang)
    text = report(data)
    (out / "compare_stats.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
