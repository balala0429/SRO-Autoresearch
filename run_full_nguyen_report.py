import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from train import (
    NGUYEN_BENCHMARKS,
    N_GRID,
    ResidualSolver,
    build_var_data,
    run_nguyen_repeated_benchmark,
)


def _safe_sympy_predict(expr_str, var_data):
    x_sym, y_sym = sp.symbols("x y")
    local_dict = {
        "x": x_sym,
        "y": y_sym,
        "relu": lambda z: sp.Max(z, 0),
    }
    expr = sp.sympify(expr_str, locals=local_dict)
    x_arr = var_data["x"]
    y_arr = var_data.get("y", x_arr)
    f = sp.lambdify((x_sym, y_sym), expr, modules=["numpy"])
    pred = f(x_arr, y_arr)
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape == ():
        pred = np.full_like(x_arr, float(pred))
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)


def _save_markdown_table(rows, out_path):
    headers = ["Dataset", "Function", "Success Rate", "Avg MSE", "Success Count", "Best Expression"]
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        md.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def _save_csv(rows, out_path):
    headers = ["Dataset", "Function", "Success Rate", "Avg MSE", "Success Count", "Best Expression"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            vals = [str(r[h]).replace('"', '""') for h in headers]
            vals = [f'"{v}"' for v in vals]
            f.write(",".join(vals) + "\n")


def _plot_one_case(case, expr_str, out_png):
    var_data = build_var_data(case, N_GRID)
    if case.get("dim", 1) == 2:
        y_true = case["target"](var_data["x"], var_data["y"])
    else:
        y_true = case["target"](var_data["x"])
    y_pred = _safe_sympy_predict(expr_str, var_data)

    plt.figure(figsize=(8, 5))
    if case.get("dim", 1) == 1:
        x = var_data["x"]
        plt.plot(x, y_true, label="Ground Truth", linewidth=2)
        plt.plot(x, y_pred, "--", label="Best Expression", linewidth=2)
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        x = var_data["x"]
        y = var_data["y"]
        plt.subplot(1, 2, 1)
        plt.tricontourf(x, y, y_true, levels=30)
        plt.title("Ground Truth")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.tricontourf(x, y, y_pred, levels=30)
        plt.title("Best Expression")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
    mse = float(np.mean((y_true - y_pred) ** 2))
    plt.suptitle(f"{case['name']} | MSE={mse:.6e}", fontsize=11)
    if case.get("dim", 1) == 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join(args.out_dir, f"plots_{tag}")
    os.makedirs(plot_dir, exist_ok=True)

    solver = ResidualSolver()
    avg_mse, n_success, n_eval, rows = run_nguyen_repeated_benchmark(
        solver,
        benchmarks=NGUYEN_BENCHMARKS,
        num_rounds=args.num_rounds,
        num_trials=args.num_trials,
        use_cache=False,
        verbose_per_case=False,
    )

    md_path = os.path.join(args.out_dir, f"nguyen_{args.num_rounds}x{args.num_trials}_{tag}.md")
    csv_path = os.path.join(args.out_dir, f"nguyen_{args.num_rounds}x{args.num_trials}_{tag}.csv")
    _save_markdown_table(rows, md_path)
    _save_csv(rows, csv_path)

    row_map = {r["Dataset"]: r for r in rows}
    for case in NGUYEN_BENCHMARKS:
        row = row_map[case["name"]]
        out_png = os.path.join(plot_dir, f"{case['name']}.png")
        _plot_one_case(case, row["Best Expression"], out_png)

    print("=" * 60)
    print(f"Saved markdown table: {md_path}")
    print(f"Saved csv table: {csv_path}")
    print(f"Saved 12 plots dir: {plot_dir}")
    print(f"Overall avg MSE: {avg_mse:.6f} | success: {n_success}/{n_eval}")
    print("=" * 60)


if __name__ == "__main__":
    main()

