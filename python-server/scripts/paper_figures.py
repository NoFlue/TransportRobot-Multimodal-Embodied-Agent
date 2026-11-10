"""논문용 그림과 표: 실험 폴더의 *_trials.csv로 영역별 성공률·완료 시간·단계 완료율을 그린다.

사용: <python with matplotlib, numpy, scipy> scripts/paper_figures.py experiments/<폴더>
출력: <폴더>/paper/ 아래 그림(PNG 300 dpi, PDF, SVG)과 표(Markdown, CSV, LaTeX). 한국어(ko)·영어(en) 두 벌.

통계
- 성공률: 성공 수 / 시행 수, 95% Wilson 구간
- 완료 시간: 성공한 시행만. 평균 ± 표준편차, 평균의 95% t 구간, 중앙값(IQR)
- 단계 완료율: 정상 완료한 필수 단계 합 / 필수 단계 합. 95% 구간은 시행 단위 부트스트랩(10,000회, 시드 0)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

AREAS = ["single", "double", "complex", "abstract"]
NAMES = {
    "ko": {"single": "단일 행동", "double": "이중 행동", "complex": "복합 행동", "abstract": "추상적 명령"},
    "en": {"single": "Single", "double": "Double", "complex": "Complex", "abstract": "Abstract"},
}
# 그림 가로축 이름. 한국어는 두 줄로 나눠 겹치지 않게 한다.
TICKS = {
    "ko": {"single": "단일\n행동", "double": "이중\n행동", "complex": "복합\n행동", "abstract": "추상적\n명령"},
    "en": NAMES["en"],
}
TEXT = {
    "ko": {
        "success": "성공률 (%)", "time": "완료 시간 (초, 성공 시행)", "steps": "단계 완료율 (%)", "area": "영역",
        "a": "(a) 성공률", "b": "(b) 완료 시간", "c": "(c) 단계 완료율", "na": "측정 안 함",
        "header": ["영역", "시행 수", "성공 수", "성공률 % [95% CI]", "완료 시간 평균 ± SD (초)", "평균의 95% CI (초)",
                   "중앙값 [IQR] (초)", "단계 완료율 % [95% CI]", "완료/필수 단계"],
        "note": ("성공률 구간은 Wilson 95% 신뢰구간, 완료 시간은 성공한 시행만 집계했고 평균의 구간은 t 분포 95% 신뢰구간, "
                 "단계 완료율 구간은 시행 단위 부트스트랩(10,000회) 95% 구간이다. 단일 행동은 단계 완료율을 측정하지 않았다."),
        "caption": "영역별 성공률, 완료 시간, 단계 완료율",
    },
    "en": {
        "success": "Success rate (%)", "time": "Completion time (s), successful trials", "steps": "Step completion rate (%)",
        "area": "Category", "a": "(a) Success rate", "b": "(b) Completion time", "c": "(c) Step completion", "na": "n/a",
        "header": ["Category", "Trials", "Successes", "Success rate % [95% CI]", "Time mean ± SD (s)", "Mean 95% CI (s)",
                   "Median [IQR] (s)", "Step completion % [95% CI]", "Completed/required steps"],
        "note": ("Success-rate intervals are 95% Wilson intervals. Completion time uses successful trials only; the mean interval is a "
                 "95% t interval. Step-completion intervals are 95% trial-level bootstrap intervals (10,000 resamples). "
                 "Step completion was not measured for single actions."),
        "caption": "Success rate, completion time, and step completion rate by command category",
    },
}
COLORS = {"single": "#4C72B0", "double": "#55A868", "complex": "#C44E52", "abstract": "#8172B2"}
HATCH = {"single": "", "double": "//", "complex": "xx", "abstract": ".."}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def bootstrap_ratio(done: np.ndarray, required: np.ndarray, rounds: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(done), size=(rounds, len(done)))
    ratios = done[idx].sum(axis=1) / required[idx].sum(axis=1)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def load(folder: Path) -> dict[str, dict]:
    result = {}
    for area in AREAS:
        path = folder / f"{area}_trials.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        success = np.array([r["success"] == "True" for r in rows])
        times = np.array([float(r["elapsed_seconds"]) for r in rows])[success]
        done = np.array([int(r["completed_steps"]) for r in rows])
        required = np.array([int(r["required_steps"]) for r in rows])
        n, k = len(rows), int(success.sum())
        item = {"n": n, "k": k, "rate": k / n, "rate_ci": wilson(k, n), "times": times,
                "mean": float(times.mean()), "sd": float(times.std(ddof=1)), "median": float(np.median(times)),
                "q1": float(np.percentile(times, 25)), "q3": float(np.percentile(times, 75))}
        t = stats.t.ppf(0.975, len(times) - 1) * item["sd"] / math.sqrt(len(times))
        item["mean_ci"] = (item["mean"] - t, item["mean"] + t)
        if area != "single":
            item.update(done=int(done.sum()), required=int(required.sum()), step=done.sum() / required.sum(),
                        step_ci=bootstrap_ratio(done, required))
        result[area] = item
    return result


def style(lang: str) -> None:
    plt.rcParams.update({
        "font.family": "Malgun Gothic" if lang == "ko" else "DejaVu Sans",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def panel_success(ax, data: dict, lang: str) -> None:
    areas = list(data)
    x = np.arange(len(areas))
    rates = [100 * data[a]["rate"] for a in areas]
    low = [100 * (data[a]["rate"] - data[a]["rate_ci"][0]) for a in areas]
    high = [100 * (data[a]["rate_ci"][1] - data[a]["rate"]) for a in areas]
    ax.bar(x, rates, width=0.62, color=[COLORS[a] for a in areas], hatch=[HATCH[a] for a in areas],
           edgecolor="black", linewidth=0.6, yerr=[low, high], capsize=3, error_kw={"elinewidth": 0.8})
    for i, a in enumerate(areas):
        ax.text(i, 100 * data[a]["rate_ci"][1] + 2, f"{data[a]['k']}/{data[a]['n']}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [TICKS[lang][a] for a in areas])
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel(TEXT[lang]["success"])
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


def panel_time(ax, data: dict, lang: str) -> None:
    areas = list(data)
    box = ax.boxplot([data[a]["times"] for a in areas], positions=np.arange(len(areas)), widths=0.55, patch_artist=True,
                     showfliers=False, medianprops={"color": "black", "linewidth": 1.0},
                     whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7}, boxprops={"linewidth": 0.7})
    for patch, a in zip(box["boxes"], areas):
        patch.set_facecolor(COLORS[a])
        patch.set_alpha(0.35)
    rng = np.random.default_rng(1)
    for i, a in enumerate(areas):
        values = data[a]["times"]
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(values)), values, s=5, color=COLORS[a], alpha=0.7, linewidths=0, zorder=3)
        ax.errorbar(i, data[a]["mean"], yerr=[[data[a]["mean"] - data[a]["mean_ci"][0]], [data[a]["mean_ci"][1] - data[a]["mean"]]],
                    fmt="D", color="black", markersize=3.5, capsize=3, elinewidth=0.9, zorder=4)
        ax.text(i, max(values) + 2, f"n={len(values)}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(np.arange(len(areas)), [TICKS[lang][a] for a in areas])
    ax.set_ylabel(TEXT[lang]["time"])
    ax.set_ylim(0, max(max(data[a]["times"]) for a in areas) * 1.12)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


def panel_steps(ax, data: dict, lang: str) -> None:
    areas = [a for a in data if "step" in data[a]]
    x = np.arange(len(areas))
    values = [100 * data[a]["step"] for a in areas]
    low = [100 * (data[a]["step"] - data[a]["step_ci"][0]) for a in areas]
    high = [100 * (data[a]["step_ci"][1] - data[a]["step"]) for a in areas]
    ax.bar(x, values, width=0.55, color=[COLORS[a] for a in areas], hatch=[HATCH[a] for a in areas],
           edgecolor="black", linewidth=0.6, yerr=[low, high], capsize=3, error_kw={"elinewidth": 0.8})
    for i, a in enumerate(areas):
        ax.text(i, 100 * data[a]["step_ci"][1] + 2, f"{data[a]['done']}/{data[a]['required']}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [TICKS[lang][a] for a in areas])
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel(TEXT[lang]["steps"])
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


def save(fig, path: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(path.with_suffix(suffix))
    plt.close(fig)


def figures(data: dict, out: Path, lang: str) -> None:
    style(lang)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [4, 4, 3]})
    for ax, draw, key in zip(axes, (panel_success, panel_time, panel_steps), "abc"):
        draw(ax, data, lang)
        ax.set_title(TEXT[lang][key], loc="left")
    fig.tight_layout(w_pad=1.6)
    save(fig, out / f"fig_results_{lang}")
    for draw, name in ((panel_success, "success"), (panel_time, "time"), (panel_steps, "steps")):
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        draw(ax, data, lang)
        fig.tight_layout()
        save(fig, out / f"fig_{name}_{lang}")


def rows(data: dict, lang: str) -> list[list[str]]:
    table = []
    for area, d in data.items():
        lo, hi = d["rate_ci"]
        step = (f"{100 * d['step']:.1f} [{100 * d['step_ci'][0]:.1f}, {100 * d['step_ci'][1]:.1f}]", f"{d['done']}/{d['required']}") \
            if "step" in d else (TEXT[lang]["na"], "–")
        table.append([
            NAMES[lang][area], str(d["n"]), str(d["k"]), f"{100 * d['rate']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]",
            f"{d['mean']:.2f} ± {d['sd']:.2f}", f"[{d['mean_ci'][0]:.2f}, {d['mean_ci'][1]:.2f}]",
            f"{d['median']:.2f} [{d['q1']:.2f}, {d['q3']:.2f}]", *step,
        ])
    return table


def tables(data: dict, out: Path, lang: str) -> None:
    header, body = TEXT[lang]["header"], rows(data, lang)
    with (out / f"table_results_{lang}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        csv.writer(stream).writerows([header, *body])
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in body]
    (out / f"table_results_{lang}.md").write_text("\n".join(md) + f"\n\n{TEXT[lang]['note']}\n", encoding="utf-8")
    esc = lambda s: s.replace("%", r"\%").replace("±", r"$\pm$").replace("–", "--")  # noqa: E731
    tex = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{TEXT[lang]['caption']}}}", r"\label{tab:results}",
           r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule",
           " & ".join(esc(h) for h in header) + r" \\", r"\midrule",
           *[" & ".join(esc(c) for c in r) + r" \\" for r in body],
           r"\bottomrule", r"\end{tabular}}", rf"\par\vspace{{2pt}}\footnotesize {esc(TEXT[lang]['note'])}", r"\end{table}"]
    (out / f"table_results_{lang}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    data = load(args.folder)
    out = args.folder / "paper"
    out.mkdir(exist_ok=True)
    for lang in ("ko", "en"):
        figures(data, out, lang)
        tables(data, out, lang)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
