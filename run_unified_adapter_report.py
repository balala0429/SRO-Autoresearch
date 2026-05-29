import argparse
import os
from datetime import datetime

import numpy as np

from benchmarks.unified_data_adapter import (
    load_feynman_sr_tasks,
    load_pmlb_tasks,
    load_srbench_tasks,
)
from train import ResidualSolver, SUCCESS_MSE_THRESHOLD


def _save_markdown_table(rows, out_path):
    headers = ["Dataset", "Source", "Profile", "Success Rate", "Avg MSE", "Success Count", "Best Expression"]
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        md.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def _evaluate_tasks(tasks, rounds, trials):
    solver = ResidualSolver()
    rows = []
    total = rounds * trials
    all_mse = []
    ok = 0
    for t in tasks:
        success = 0
        mse_sum = 0.0
        best_mse = float("inf")
        best_expr = "0"
        for _ in range(total):
            solver.var_data = t.var_data
            solver.x_test = t.var_data.get("x", next(iter(t.var_data.values())))
            mse, expr = solver.solve(
                y_obs=t.y_obs,
                profile=t.profile,
                suite=t.suite or t.source,
                plot=False,
                verbose=False,
            )
            mse_sum += mse
            if mse < best_mse:
                best_mse = mse
                best_expr = expr
            if mse < SUCCESS_MSE_THRESHOLD:
                success += 1
        avg_mse = mse_sum / total
        all_mse.append(avg_mse)
        ok += int(success == total)
        rows.append(
            {
                "Dataset": t.name,
                "Source": t.source,
                "Profile": t.profile,
                "Success Rate": f"{(success / total) * 100:.2f}%",
                "Avg MSE": f"{avg_mse:.6f}",
                "Success Count": f"{success}/{total}",
                "Best Expression": best_expr,
            }
        )
        print(f"{t.name}: success_rate={(success / total) * 100:.2f}% | avg_mse={avg_mse:.6f}")
    return rows, float(np.mean(all_mse) if all_mse else np.nan), ok, len(tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, choices=["pmlb", "srbench", "feynman"], default="srbench")
    parser.add_argument("--datasets", type=str, default="", help="仅用于 source=pmlb，逗号分隔")
    parser.add_argument("--max_count", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()

    if args.source == "pmlb":
        ds = [x.strip() for x in args.datasets.split(",") if x.strip()]
        if not ds:
            raise ValueError("--source pmlb 时必须给 --datasets")
        tasks = load_pmlb_tasks(ds)
    elif args.source == "srbench":
        tasks = load_srbench_tasks(max_count=args.max_count)
    else:
        tasks = load_feynman_sr_tasks(max_count=args.max_count)

    os.makedirs(args.out_dir, exist_ok=True)
    rows, avg_mse, n_success, n_total = _evaluate_tasks(tasks, args.num_rounds, args.num_trials)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(
        args.out_dir,
        f"{args.source}_unified_{args.num_rounds}x{args.num_trials}_{tag}.md",
    )
    _save_markdown_table(rows, out)
    print("=" * 60)
    print(f"saved: {out}")
    print(f"overall avg MSE: {avg_mse:.6f} | full-success datasets: {n_success}/{n_total}")
    print("=" * 60)


if __name__ == "__main__":
    main()

