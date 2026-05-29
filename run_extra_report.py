import argparse
import os
from datetime import datetime

from benchmarks.extra_symbolic_benchmarks import EXTRA_BENCHMARKS
from train import ResidualSolver, run_nguyen_repeated_benchmark


def _save_markdown_table(rows, out_path):
    headers = ["Dataset", "Function", "Success Rate", "Avg MSE", "Success Count", "Best Expression"]
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        md.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--num_trials", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    solver = ResidualSolver()
    avg_mse, n_success, n_eval, rows = run_nguyen_repeated_benchmark(
        solver,
        benchmarks=EXTRA_BENCHMARKS,
        num_rounds=args.num_rounds,
        num_trials=args.num_trials,
        use_cache=False,
        verbose_per_case=False,
    )

    md_path = os.path.join(args.out_dir, f"extra_bench_{args.num_rounds}x{args.num_trials}_{tag}.md")
    _save_markdown_table(rows, md_path)

    print("=" * 60)
    print(f"Saved extra benchmark table: {md_path}")
    print(f"Overall avg MSE: {avg_mse:.6f} | success: {n_success}/{n_eval}")
    print("=" * 60)


if __name__ == "__main__":
    main()

