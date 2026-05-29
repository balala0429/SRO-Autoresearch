import argparse
import csv
import glob
import os
import subprocess
from datetime import datetime


def run_cmd(cmd, cwd):
    print(f"[RUN] {cmd}")
    p = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def newest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def summarize_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return {"datasets": 0, "avg_mse": float("nan"), "full_success": 0}
    avg_mse = sum(float(r["Avg MSE"]) for r in rows) / len(rows)
    full_success = sum(1 for r in rows if r["Success Rate"] == "100.00%")
    return {"datasets": len(rows), "avg_mse": avg_mse, "full_success": full_success}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_cmd(
        "python run_benchmark_platform.py --source nguyen --num_rounds 10 --num_trials 100",
        root,
    )
    ng_csv = newest_file(os.path.join(root, "reports", "nguyen_10x100_*.csv"))
    ng_md = newest_file(os.path.join(root, "reports", "nguyen_10x100_*.md"))
    ng_plot = newest_file(os.path.join(root, "reports", "plots_nguyen_*"))

    run_cmd(
        "python run_benchmark_platform.py --source feynman --max_count 999 --num_rounds 5 --num_trials 30",
        root,
    )
    fy_csv = newest_file(os.path.join(root, "reports", "feynman_5x30_*.csv"))
    fy_md = newest_file(os.path.join(root, "reports", "feynman_5x30_*.md"))
    fy_plot = newest_file(os.path.join(root, "reports", "plots_feynman_*"))

    run_cmd(
        "python run_benchmark_platform.py --source srbench --max_count 53 --num_rounds 1 --num_trials 30",
        root,
    )
    sr_csv = newest_file(os.path.join(root, "reports", "srbench_1x30_*.csv"))
    sr_md = newest_file(os.path.join(root, "reports", "srbench_1x30_*.md"))
    sr_plot = newest_file(os.path.join(root, "reports", "plots_srbench_*"))

    ng = summarize_csv(ng_csv)
    fy = summarize_csv(fy_csv)
    sr = summarize_csv(sr_csv)

    summary_path = os.path.join(root, args.out_dir, f"formal_matrix_summary_{tag}.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Formal Experiment Matrix Summary\n\n")
        f.write(f"- Generated at: {tag}\n\n")
        f.write("## Matrix\n")
        f.write("- Nguyen: 10x100\n")
        f.write("- Feynman: full tasks x 5x30\n")
        f.write("- SRBench: 53 tasks x 1x30\n\n")
        f.write("## Results Overview\n\n")
        f.write("| Suite | Datasets | Avg MSE | 100% Success Datasets |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| Nguyen | {ng['datasets']} | {ng['avg_mse']:.6f} | {ng['full_success']} |\n")
        f.write(f"| Feynman | {fy['datasets']} | {fy['avg_mse']:.6f} | {fy['full_success']} |\n")
        f.write(f"| SRBench | {sr['datasets']} | {sr['avg_mse']:.6f} | {sr['full_success']} |\n\n")

        f.write("## Artifacts\n\n")
        f.write(f"- Nguyen table (md): `{os.path.relpath(ng_md, root)}`\n")
        f.write(f"- Nguyen table (csv): `{os.path.relpath(ng_csv, root)}`\n")
        f.write(f"- Nguyen plots: `{os.path.relpath(ng_plot, root)}`\n\n")
        f.write(f"- Feynman table (md): `{os.path.relpath(fy_md, root)}`\n")
        f.write(f"- Feynman table (csv): `{os.path.relpath(fy_csv, root)}`\n")
        f.write(f"- Feynman plots: `{os.path.relpath(fy_plot, root)}`\n\n")
        f.write(f"- SRBench table (md): `{os.path.relpath(sr_md, root)}`\n")
        f.write(f"- SRBench table (csv): `{os.path.relpath(sr_csv, root)}`\n")
        f.write(f"- SRBench plots: `{os.path.relpath(sr_plot, root)}`\n")

    print("=" * 60)
    print(f"Summary written to: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

