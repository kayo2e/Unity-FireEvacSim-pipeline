"""
exp3_visualize.py — 실험 ③: 전체 결과 비교 테이블 + 차트 생성
==============================================================
result/ 하위 모든 JSON 요약 파일을 읽어 비교 테이블 출력 및 차트 저장.

실행:
    cd stage2
    python experiments/exp3_visualize.py
    python experiments/exp3_visualize.py --no-chart   # 차트 없이 텍스트만
    python experiments/exp3_visualize.py --result-dir result
"""

import sys, os, json, argparse, glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

RESULT_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "result")
CHART_DIR   = os.path.join(RESULT_BASE, "charts")

SCENARIO_NAMES = {1: "기본탈출", 2: "EXIT A 위협", 3: "진입로차단", 4: "양방향위협"}
MODEL_ORDER = ["astar", "ppo", "recurrent_ppo", "joint_ppo", "autoregressive_ppo"]
MODEL_LABELS = {
    "astar":              "A* 베이스라인",
    "ppo":                "표준 PPO",
    "recurrent_ppo":      "RecurrentPPO",
    "joint_ppo":          "JointPPO",
    "autoregressive_ppo": "AutoregressivePPO",
}


# ── JSON 결과 로드 ─────────────────────────────
def load_summaries(result_dir: str) -> dict:
    """
    result/{model}/test_summary_*.json 파일을 읽어
    {model_name: {scenario: summary_dict}} 구조로 반환.
    """
    data = {}
    pattern = os.path.join(result_dir, "**", "test_summary_*.json")
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            continue
        model = _infer_model(path, s)
        sc    = s.get("scenario")
        if model and sc:
            data.setdefault(model, {})
            # 같은 시나리오 파일이 여러 개면 가장 최근 것 사용
            existing = data[model].get(sc)
            if existing is None or path > existing.get("_path", ""):
                s["_path"] = path
                data[model][sc] = s
    return data


def _infer_model(path: str, summary: dict) -> str:
    """파일 경로 또는 summary["model"] 필드에서 모델명 추론."""
    m = summary.get("model", "")
    for key in MODEL_ORDER:
        if key in path or key in m:
            return key
    return None


# ── exp1 결과 로드 (별도 포맷) ────────────────
def load_exp1(result_dir: str) -> dict:
    """
    result/exp1_compare/exp1_*_summary.json 에서 astar / ppo 비교 데이터 로드.
    {scenario: {"astar": {...}, "ppo": {...}}} 반환.
    """
    data = {}
    pattern = os.path.join(result_dir, "exp1_compare", "exp1_*_summary.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            continue
        sc = s.get("scenario")
        if sc:
            data[sc] = s
    return data


def load_exp2(result_dir: str) -> dict:
    """
    result/exp2_blocking/exp2_*_summary.json 에서 blocking 결과 로드.
    {scenario: [{"block_exit":..,"astar":{..},"ppo":{..}}, ...]} 반환.
    """
    data = {}
    pattern = os.path.join(result_dir, "exp2_blocking", "exp2_*_summary.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            continue
        sc = s.get("scenario")
        if sc:
            data.setdefault(sc, []).append(s)
    return data


# ── 비교 테이블 출력 ──────────────────────────
def print_main_table(data: dict):
    """모델 × 시나리오 생존율 비교 테이블."""
    scenarios = sorted({sc for m in data.values() for sc in m.keys()})
    models    = [m for m in MODEL_ORDER if m in data]

    if not models:
        print("  [결과 없음] result/ 디렉터리에 test_summary_*.json 파일이 없습니다.")
        return

    print(f"\n{'═'*72}")
    print("  모델별 생존율 비교  (mean ± std)")
    print(f"{'─'*72}")
    header = f"  {'모델':<22}"
    for sc in scenarios:
        header += f"  S{sc} {SCENARIO_NAMES.get(sc, ''):<10}"
    print(header)
    print(f"{'─'*72}")

    for model in models:
        label = MODEL_LABELS.get(model, model)
        row = f"  {label:<22}"
        for sc in scenarios:
            s = data[model].get(sc, {})
            sr = s.get("survival_rate", {})
            if sr:
                row += f"  {sr['mean']:.1%}±{sr['std']:.1%}  "
            else:
                row += f"  {'N/A':>12}  "
        print(row)
    print(f"{'═'*72}")


def print_exp1_table(exp1: dict):
    if not exp1:
        return
    print(f"\n{'═'*72}")
    print("  실험① 결과: A* vs PPO 직접 비교 (생존율 mean)")
    print(f"{'─'*72}")
    print(f"  {'시나리오':<18} {'A* 베이스라인':>14}  {'PPO':>12}  {'향상폭':>8}")
    print(f"{'─'*72}")
    for sc in sorted(exp1.keys()):
        d    = exp1[sc]
        a_sr = d.get("astar", {}).get("survival_rate", {}).get("mean", float("nan"))
        p_sr = d.get("ppo",   {}).get("survival_rate", {}).get("mean", float("nan"))
        diff = p_sr - a_sr if not (np.isnan(a_sr) or np.isnan(p_sr)) else float("nan")
        arrow = ("▲" if diff > 0 else "▼") if not np.isnan(diff) else ""
        name = SCENARIO_NAMES.get(sc, f"S{sc}")
        print(f"  S{sc} {name:<14} {a_sr:>12.1%}  {p_sr:>12.1%}  "
              f"{arrow}{abs(diff):.1%}" if not np.isnan(diff) else
              f"  S{sc} {name:<14} {a_sr:>12.1%}  {'N/A':>12}")
    print(f"{'═'*72}")


def print_exp2_table(exp2: dict):
    if not exp2:
        return
    print(f"\n{'═'*72}")
    print("  실험② 결과: 중간 출구 차단 스트레스 테스트")
    print(f"{'─'*72}")
    print(f"  {'시나리오':<14} {'차단출구':>6} {'@스텝':>6}  "
          f"{'A* 생존율':>10}  {'PPO 생존율':>10}  {'향상폭':>8}")
    print(f"{'─'*72}")
    for sc in sorted(exp2.keys()):
        for entry in exp2[sc]:
            block_exit = entry.get("block_exit", "?")
            block_step = entry.get("block_at_step", "?")
            a_sr = entry.get("astar", {}).get("survival_rate", {}).get("mean", float("nan"))
            p_sr = entry.get("ppo",   {}).get("survival_rate", {}).get("mean", float("nan"))
            diff = p_sr - a_sr if not (np.isnan(a_sr) or np.isnan(p_sr)) else float("nan")
            arrow = ("▲" if diff > 0 else "▼") if not np.isnan(diff) else ""
            name = SCENARIO_NAMES.get(sc, f"S{sc}")
            if np.isnan(diff):
                print(f"  S{sc} {name:<10} {block_exit:>6} {block_step:>6}  "
                      f"{a_sr:>10.1%}  {'N/A':>10}")
            else:
                print(f"  S{sc} {name:<10} {block_exit:>6} {block_step:>6}  "
                      f"{a_sr:>10.1%}  {p_sr:>10.1%}  {arrow}{abs(diff):.1%}")
    print(f"{'═'*72}")


# ── 차트 생성 ──────────────────────────────────
def make_charts(data: dict, exp1: dict, exp2: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        # 한글 폰트 시도
        for font in ["NanumGothic", "AppleGothic", "DejaVu Sans"]:
            if any(font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
                plt.rcParams["font.family"] = font
                break
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("  [경고] matplotlib 미설치 — 차트 생략")
        return

    os.makedirs(CHART_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    scenarios = sorted({sc for m in data.values() for sc in m.keys()}) or [1, 2, 3, 4]
    models    = [m for m in MODEL_ORDER if m in data]

    # ── 차트 1: 시나리오별 생존율 막대그래프 ─────
    if models and scenarios:
        fig, axes = plt.subplots(1, len(scenarios),
                                  figsize=(4 * len(scenarios), 5), sharey=True)
        if len(scenarios) == 1:
            axes = [axes]
        colors = ["#5B9BD5", "#ED7D31", "#70AD47", "#FFC000", "#7030A0"]

        for ax, sc in zip(axes, scenarios):
            means = []
            stds  = []
            labels = []
            clrs  = []
            for i, model in enumerate(models):
                sr = data[model].get(sc, {}).get("survival_rate", {})
                if sr:
                    means.append(sr["mean"])
                    stds.append(sr["std"])
                    labels.append(MODEL_LABELS.get(model, model).replace(" ", "\n"))
                    clrs.append(colors[i % len(colors)])
            if means:
                bars = ax.bar(range(len(means)), means, color=clrs, alpha=0.85,
                               yerr=stds, capsize=4, error_kw={"linewidth": 1})
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=7)
                ax.set_title(f"S{sc}\n{SCENARIO_NAMES.get(sc, '')}", fontsize=9)
                ax.set_ylim(0, 1.05)
                for bar, mean in zip(bars, means):
                    ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                            f"{mean:.0%}", ha="center", va="bottom", fontsize=7)
            ax.set_ylabel("생존율" if ax == axes[0] else "")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("시나리오별 모델 생존율 비교", fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(CHART_DIR, f"chart1_survival_{ts}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  차트 저장: {path}")

    # ── 차트 2: 출구 분배 (escaped_A vs escaped_B) ─
    if models and scenarios:
        fig, axes = plt.subplots(1, len(scenarios),
                                  figsize=(4 * len(scenarios), 5))
        if len(scenarios) == 1:
            axes = [axes]

        for ax, sc in zip(axes, scenarios):
            model_labels, a_vals, b_vals = [], [], []
            for model in models:
                s  = data[model].get(sc, {})
                ea = s.get("escaped_A", {}).get("mean")
                eb = s.get("escaped_B", {}).get("mean")
                if ea is not None and eb is not None:
                    model_labels.append(MODEL_LABELS.get(model, model).replace(" ", "\n"))
                    a_vals.append(ea)
                    b_vals.append(eb)
            if model_labels:
                x = np.arange(len(model_labels))
                w = 0.35
                ax.bar(x - w/2, a_vals, w, label="출구 A (4셀)", color="#5B9BD5", alpha=0.85)
                ax.bar(x + w/2, b_vals, w, label="출구 B (2셀)", color="#ED7D31", alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(model_labels, fontsize=7)
                ax.set_title(f"S{sc} 출구 분배", fontsize=9)
                ax.legend(fontsize=7)
                ax.grid(axis="y", alpha=0.3)

        fig.suptitle("모델별 출구 탈출 분배", fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(CHART_DIR, f"chart2_exit_dist_{ts}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  차트 저장: {path}")

    # ── 차트 3: 실험② 차단 스트레스 결과 ──────
    if exp2:
        scenarios_with_data = sorted(exp2.keys())
        fig, axes = plt.subplots(1, len(scenarios_with_data),
                                  figsize=(5 * len(scenarios_with_data), 5))
        if len(scenarios_with_data) == 1:
            axes = [axes]

        for ax, sc in zip(axes, scenarios_with_data):
            entries = exp2[sc]
            labels, a_bars, p_bars = [], [], []
            for e in entries:
                a_sr = e.get("astar", {}).get("survival_rate", {}).get("mean")
                p_sr = e.get("ppo",   {}).get("survival_rate", {}).get("mean")
                if a_sr is not None:
                    lbl = f"차단{e['block_exit']}@{e['block_at_step']}"
                    labels.append(lbl)
                    a_bars.append(a_sr)
                    p_bars.append(p_sr if p_sr is not None else 0)

            if labels:
                x = np.arange(len(labels))
                w = 0.35
                ax.bar(x - w/2, a_bars, w, label="A* 베이스라인",
                        color="#5B9BD5", alpha=0.85)
                ax.bar(x + w/2, p_bars, w, label="PPO",
                        color="#ED7D31", alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
                ax.set_title(f"S{sc} {SCENARIO_NAMES.get(sc, '')} — 차단 테스트",
                              fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(axis="y", alpha=0.3)

        fig.suptitle("출구 차단 스트레스: A* vs PPO 생존율", fontsize=12,
                      fontweight="bold")
        plt.tight_layout()
        path = os.path.join(CHART_DIR, f"chart3_blocking_{ts}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  차트 저장: {path}")


# ── 진입점 ────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실험③: 전체 결과 시각화")
    parser.add_argument("--result-dir", type=str, default=RESULT_BASE)
    parser.add_argument("--no-chart",   action="store_true")
    args = parser.parse_args()

    print(f"\n결과 디렉터리 스캔: {args.result_dir}")
    data = load_summaries(args.result_dir)
    exp1 = load_exp1(args.result_dir)
    exp2 = load_exp2(args.result_dir)

    print(f"  발견된 모델: {list(data.keys())}")
    print(f"  발견된 exp1 시나리오: {list(exp1.keys())}")
    print(f"  발견된 exp2 시나리오: {list(exp2.keys())}")

    print_main_table(data)
    print_exp1_table(exp1)
    print_exp2_table(exp2)

    if not args.no_chart:
        print(f"\n차트 생성 중...")
        make_charts(data, exp1, exp2)
        print(f"차트 저장 위치: {CHART_DIR}")
