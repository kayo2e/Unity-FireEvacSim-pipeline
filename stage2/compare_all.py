"""
결과 취합 비교 (Dijkstra / A* / RL)
======================================
각 eval_*.py가 생성한 result_*.json을 읽어서
비교 테이블 + CSV 출력.

실행: python compare_all.py [--people 10] [--csv comparison.csv]

사전 실행:
  python eval_dijkstra.py
  python eval_astar.py
  python eval_rl.py
"""

import sys, json, glob
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from train import SCENARIO_CONFIGS


def load_results(people):
    entries = []

    for path in sorted(glob.glob(f"result_*_{people}ppl.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        strategy = data["strategy"]
        results = data["results"]

        for sc_str, records in results.items():
            entries.append((strategy, sc_str, records))

    return entries


def build_table(entries):
    table = {}
    for strategy, sc_str, records in entries:
        sr   = [r["survival_rate"] for r in records]
        esc  = [r["escaped"] for r in records]
        dead = [r["dead"] for r in records]
        steps = [r["steps"] for r in records]
        table[(strategy, sc_str)] = {
            "mean_sr": np.mean(sr), "std_sr": np.std(sr),
            "mean_esc": np.mean(esc), "mean_dead": np.mean(dead),
            "mean_steps": np.mean(steps),
        }
    return table


def print_comparison(table):
    scenarios = sorted(set(sc for _, sc in table.keys()), key=int)
    strategies = sorted(set(st for st, _ in table.keys()))

    for sc in scenarios:
        sc_name = SCENARIO_CONFIGS[int(sc)]["name"]
        print(f"\n{'='*70}")
        print(f"  시나리오 {sc} ({sc_name})")
        print(f"{'='*70}")
        print(f"  {'전략':<22} {'생존율':>8} {'±std':>7} {'탈출':>6} {'사망':>6} {'스텝':>6}")
        print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")

        rows = []
        for st in strategies:
            key = (st, sc)
            if key in table:
                rows.append((st, table[key]))
        rows.sort(key=lambda x: x[1]["mean_sr"], reverse=True)

        for rank, (st, d) in enumerate(rows, 1):
            marker = " <-- BEST" if rank == 1 else ""
            print(f"  {st:<22} {d['mean_sr']:>7.1%} {d['std_sr']:>6.3f} "
                  f"{d['mean_esc']:>6.1f} {d['mean_dead']:>6.1f} "
                  f"{d['mean_steps']:>6.1f}{marker}")

    # 전체 요약
    print(f"\n\n{'='*70}")
    print(f"  전체 요약 - 시나리오별 평균 생존율")
    print(f"{'='*70}")

    header = f"  {'전략':<22}"
    for sc in scenarios:
        header += f" {'S'+sc:>10}"
    header += f" {'전체평균':>10}"
    print(header)
    print(f"  {'-'*22}" + f" {'-'*10}" * (len(scenarios) + 1))

    summary = []
    for st in strategies:
        vals = []
        row = f"  {st:<22}"
        for sc in scenarios:
            key = (st, sc)
            if key in table:
                v = table[key]["mean_sr"]
                vals.append(v)
                row += f" {v:>9.1%}"
            else:
                row += f" {'N/A':>10}"
        avg = np.mean(vals) if vals else 0
        row += f" {avg:>9.1%}"
        summary.append((st, avg, row))

    summary.sort(key=lambda x: x[1], reverse=True)
    for _, _, row in summary:
        print(row)


def save_csv(table, output_path):
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "scenario", "scenario_name",
            "mean_survival_rate", "std_survival_rate",
            "mean_escaped", "mean_dead", "mean_steps",
        ])
        for (st, sc), d in sorted(table.items()):
            sc_name = SCENARIO_CONFIGS[int(sc)]["name"]
            writer.writerow([
                st, sc, sc_name,
                f"{d['mean_sr']:.4f}", f"{d['std_sr']:.4f}",
                f"{d['mean_esc']:.2f}", f"{d['mean_dead']:.2f}",
                f"{d['mean_steps']:.2f}",
            ])
    print(f"\n[저장] CSV: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dijkstra / A* / RL 비교")
    parser.add_argument("--people", type=int, default=10)
    parser.add_argument("--csv",    type=str, default="comparison.csv")
    args = parser.parse_args()

    entries = load_results(args.people)
    if not entries:
        print(f"[!] result_*_{args.people}ppl.json 파일이 없습니다.")
        print("    먼저 각 eval_*.py를 실행하세요:")
        print("      python eval_dijkstra.py")
        print("      python eval_astar.py")
        print("      python eval_rl.py")
        sys.exit(1)

    strats = set(st for st, _, _ in entries)
    print(f"로드된 전략: {len(strats)}개 - {', '.join(sorted(strats))}")

    table = build_table(entries)
    print_comparison(table)
    save_csv(table, args.csv)
