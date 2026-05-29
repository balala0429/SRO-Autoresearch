import argparse
import os
from datetime import datetime
from typing import List

import numpy as np
import sympy as sp

from benchmarks.extra_symbolic_benchmarks import EXTRA_BENCHMARKS
from benchmarks.unified_data_adapter import (
    UnifiedTask,
    load_feynman_sr_tasks,
    load_pmlb_tasks,
    load_srbench_tasks,
)
from train import (
    NGUYEN_BENCHMARKS,
    N_GRID,
    ResidualSolver,
    SUCCESS_MSE_THRESHOLD,
    build_var_data,
    resolve_suite,
)

SOURCE_BUDGET_DEFAULTS = {
    "nguyen": (10, 100),
    "extra": (2, 25),
    "feynman": (2, 25),
    "pmlb": (1, 20),
    "srbench": (1, 20),
}


def _safe_sympy_predict(expr_str: str, var_data: dict) -> np.ndarray:
    x_sym, y_sym = sp.symbols("x y")
    expr = sp.sympify(expr_str, locals={"x": x_sym, "y": y_sym, "relu": lambda z: sp.Max(z, 0)})
    x_arr = var_data["x"]
    y_arr = var_data.get("y", x_arr)
    f = sp.lambdify((x_sym, y_sym), expr, modules=["numpy"])
    pred = f(x_arr, y_arr)
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape == ():
        pred = np.full_like(x_arr, float(pred))
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)


def _save_markdown_table(rows, out_path):
    headers = ["Dataset", "Source", "Function", "Profile", "Success Rate", "Avg MSE", "Success Count", "Best Expression"]
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        md.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def _save_csv(rows, out_path):
    headers = ["Dataset", "Source", "Function", "Profile", "Success Rate", "Avg MSE", "Success Count", "Best Expression"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            vals = [str(r[h]).replace('"', '""') for h in headers]
            vals = [f'"{v}"' for v in vals]
            f.write(",".join(vals) + "\n")


def _build_cases(source: str, datasets: str, max_count: int):
    def _from_builtin(case, src_name: str):
        def _get_data():
            var_data = build_var_data(case, N_GRID)
            if case.get("dim", 1) == 2:
                y_obs = case["target"](var_data["x"], var_data["y"])
            else:
                y_obs = case["target"](var_data["x"])
            return var_data, y_obs

        return {
            "name": case["name"],
            "source": src_name,
            "suite": case.get("suite", src_name),
            "function_str": case.get("function_str", ""),
            "profile": case["profile"],
            "get_data": _get_data,
        }

    if source == "nguyen":
        return [_from_builtin(c, "nguyen") for c in NGUYEN_BENCHMARKS]
    if source == "extra":
        return [_from_builtin(c, "extra") for c in EXTRA_BENCHMARKS]
    if source == "pmlb":
        ds = [x.strip() for x in datasets.split(",") if x.strip()]
        if not ds:
            raise ValueError("--source pmlb 时必须给 --datasets")
        tasks: List[UnifiedTask] = load_pmlb_tasks(ds)
    elif source == "srbench":
        tasks = load_srbench_tasks(max_count=max_count)
    elif source == "feynman":
        tasks = load_feynman_sr_tasks(max_count=max_count)
    else:
        raise ValueError(f"unknown source: {source}")

    return [
        {
            "name": t.name,
            "source": t.source,
            "suite": t.suite or ("feynman" if t.source.startswith("feynman") else t.source),
            "function_str": t.function_str,
            "profile": t.profile,
            "get_data": lambda task=t: (task.var_data, task.y_obs),
        }
        for t in tasks
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, choices=["nguyen", "extra", "pmlb", "srbench", "feynman"], default="nguyen")
    parser.add_argument("--datasets", type=str, default="", help="仅用于 source=pmlb，逗号分隔")
    parser.add_argument("--max_count", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=None)
    parser.add_argument("--num_trials", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()

    budget = SOURCE_BUDGET_DEFAULTS.get(args.source, (1, 20))
    if args.num_rounds is None:
        args.num_rounds = budget[0]
    if args.num_trials is None:
        args.num_trials = budget[1]

    cases = _build_cases(args.source, args.datasets, args.max_count)
    if not cases:
        raise RuntimeError("没有可运行的数据集，请检查 source/datasets 配置")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plotting disabled per request: only export tables
    plot_dir = None

    solver = ResidualSolver()
    total_trials = args.num_rounds * args.num_trials
    rows = []
    avg_mses = []
    full_success = 0

    print("\n" + "=" * 60)
    print(f"📋 Unified Benchmark | source={args.source} | rounds={args.num_rounds} trials/round={args.num_trials}")
    print("=" * 60)

    for case in cases:
        var_data, y_obs = case["get_data"]()
        success_count = 0
        mse_sum = 0.0
        best_mse = float("inf")
        best_expr = "0"
        for _ in range(total_trials):
            solver.var_data = var_data
            solver.x_test = var_data.get("x", next(iter(var_data.values())))
            mse, expr = solver.solve(
                y_obs=y_obs,
                profile=case["profile"],
                suite=case.get("suite") or resolve_suite(case),
                plot=False,
                verbose=False,
            )
            mse_sum += mse
            if mse < best_mse:
                best_mse = mse
                best_expr = expr
            if mse < SUCCESS_MSE_THRESHOLD:
                success_count += 1
        avg_mse = mse_sum / total_trials
        success_rate = success_count / total_trials
        rows.append(
            {
                "Dataset": case["name"],
                "Source": case["source"],
                "Function": case["function_str"],
                "Profile": case["profile"],
                "Success Rate": f"{success_rate * 100:.2f}%",
                "Avg MSE": f"{avg_mse:.6f}",
                "Success Count": f"{success_count}/{total_trials}",
                "Best Expression": best_expr,
            }
        )
        avg_mses.append(avg_mse)
        full_success += int(success_count == total_trials)
        print(f"{case['name']}: success_rate={success_rate * 100:.2f}% | avg_mse={avg_mse:.6f}")

    md_path = os.path.join(args.out_dir, f"{args.source}_{args.num_rounds}x{args.num_trials}_{tag}.md")
    csv_path = os.path.join(args.out_dir, f"{args.source}_{args.num_rounds}x{args.num_trials}_{tag}.csv")
    _save_markdown_table(rows, md_path)
    _save_csv(rows, csv_path)

    print("=" * 60)
    print(f"Saved markdown table: {md_path}")
    print(f"Saved csv table: {csv_path}")
    print("Saved plots dir: disabled")
    print(f"Overall avg MSE: {float(np.mean(avg_mses)):.6f} | full-success datasets: {full_success}/{len(cases)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

