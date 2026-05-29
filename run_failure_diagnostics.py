"""
对 Extra / Feynman 失败任务做根因诊断：
- prior_only：仅靠结构先验能否拟合
- no_bootstrap：关闭 bootstrap 的完整求解
- suite_default：数据集感知 Full Model
- legacy_full：强制 full bootstrap（旧 Full 行为）
"""
import argparse
import copy
from datetime import datetime
import os
from statistics import mean, pstdev

from benchmarks.extra_symbolic_benchmarks import EXTRA_BENCHMARKS
from benchmarks.unified_data_adapter import load_feynman_sr_tasks
from tool.tree_utils import is_valid_tree
from train import (
    PROFILE_CONFIG,
    ResidualSolver,
    SUCCESS_MSE_THRESHOLD,
    build_var_data,
    resolve_suite,
)


def _classify_failure(diag, final_mse):
    prior_mse = diag.get("prior_only_mse", float("inf"))
    boot_mse = diag.get("bootstrap_mse")
    max_sim = diag.get("max_faiss_similarity")
    if final_mse < SUCCESS_MSE_THRESHOLD:
        return "ok"
    if prior_mse < SUCCESS_MSE_THRESHOLD and (boot_mse is None or boot_mse >= SUCCESS_MSE_THRESHOLD):
        return "prior_ok_but_pipeline_degraded"
    if prior_mse >= SUCCESS_MSE_THRESHOLD and (max_sim is None or max_sim < 0.25):
        return "weak_retrieval_signal"
    if prior_mse >= SUCCESS_MSE_THRESHOLD and max_sim is not None and max_sim >= 0.25:
        return "structure_mismatch_or_splice"
    if boot_mse is not None and boot_mse < SUCCESS_MSE_THRESHOLD and final_mse >= SUCCESS_MSE_THRESHOLD:
        return "bootstrap_ok_iter_failed"
    dim = diag.get("input_points", 127)
    if dim == 32 * 32:
        return "possible_2d_grid_or_feature_issue"
    return "unknown"


def _run_modes(solver, y_obs, profile, suite, var_data):
    solver.var_data = var_data
    solver.x_test = var_data.get("x", next(iter(var_data.values())))
    prior_trees = [t for t in ResidualSolver._build_prior_trees(profile) if is_valid_tree(t)]

    rows = []
    modes = [
        ("prior_only_mse", None),
        ("legacy_full", "nguyen"),
        ("suite_default", suite),
        ("no_bootstrap", suite),
    ]
    for label, suite_override in modes:
        if label == "prior_only_mse":
            mse = ResidualSolver.prior_only_mse(solver, prior_trees, y_obs, var_data)
            rows.append((label, mse, {}))
            continue
        backup = copy.deepcopy(PROFILE_CONFIG)
        try:
            if label == "no_bootstrap":
                for cfg in PROFILE_CONFIG.values():
                    cfg["bootstrap_priors"] = False
                    cfg["bootstrap_mode"] = "off"
            mse, _ = solver.solve(
                y_obs=y_obs,
                profile=profile,
                suite=suite_override if suite_override else suite,
                plot=False,
                verbose=False,
                record_diagnostics=True,
            )
            rows.append((label, mse, dict(solver.last_diagnostic)))
        finally:
            PROFILE_CONFIG.clear()
            PROFILE_CONFIG.update(backup)
    return rows


def _iter_cases(source, max_feynman):
    if source in ("extra", "all"):
        for c in EXTRA_BENCHMARKS:
            yield c
    if source in ("feynman", "all"):
        for t in load_feynman_sr_tasks(max_count=max_feynman):
            yield {
                "name": t.name,
                "dim": t.dim,
                "profile": t.profile,
                "suite": t.suite or "feynman",
                "source": t.source,
                "function_str": t.function_str,
                "_var_data": t.var_data,
                "_y_obs": t.y_obs,
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["extra", "feynman", "all"], default="all")
    parser.add_argument("--max_feynman", type=int, default=8)
    parser.add_argument("--num_trials", type=int, default=3, help="每任务重复次数（取 best MSE）")
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"failure_diagnostics_{tag}.md")

    solver = ResidualSolver()
    lines = [
        "# Failure Diagnostics",
        "",
        f"- source: {args.source}",
        f"- trials per task (best of): {args.num_trials}",
        f"- success threshold: MSE < {SUCCESS_MSE_THRESHOLD}",
        "",
        "| Task | Suite | Profile | Dim | Mode | Best MSE | Failure Class | Notes |",
        "|---|---|---|---:|---|---:|---|---|",
    ]

    for case in _iter_cases(args.source, args.max_feynman):
        if "_y_obs" in case:
            var_data, y_obs = case["_var_data"], case["_y_obs"]
            dim = case["dim"]
        else:
            var_data = build_var_data(case, 127)
            dim = case.get("dim", 1)
            if dim == 2:
                y_obs = case["target"](var_data["x"], var_data["y"])
            else:
                y_obs = case["target"](var_data["x"])

        suite = resolve_suite(case)
        profile = case["profile"]
        best_by_mode = {}
        diag_by_mode = {}
        for _ in range(args.num_trials):
            for label, mse, diag in _run_modes(solver, y_obs, profile, suite, var_data):
                if label not in best_by_mode or mse < best_by_mode[label]:
                    best_by_mode[label] = mse
                    diag_by_mode[label] = diag

        final_mse = best_by_mode.get("suite_default", float("inf"))
        failure_class = _classify_failure(diag_by_mode.get("suite_default", {}), final_mse)
        for label, mse in sorted(best_by_mode.items()):
            notes = ""
            d = diag_by_mode.get(label, {})
            if d:
                parts = []
                if d.get("bootstrap_mse") is not None:
                    parts.append(f"boot={d['bootstrap_mse']:.2e}")
                if d.get("max_faiss_similarity") is not None:
                    parts.append(f"sim={d['max_faiss_similarity']:.3f}")
                if d.get("n_active_terms"):
                    parts.append(f"terms={d['n_active_terms']}")
                notes = ", ".join(parts)
            lines.append(
                f"| {case['name']} | {suite} | {profile} | {dim} | {label} | {mse:.6e} | "
                f"{failure_class if label == 'suite_default' else ''} | {notes} |"
            )

        # 对比 legacy vs suite
        legacy = best_by_mode.get("legacy_full", float("inf"))
        suite_m = best_by_mode.get("suite_default", float("inf"))
        if legacy + 1e-12 < suite_m:
            lines.append(f"\n> **{case['name']}**: legacy_full 优于 suite_default，可调低 `{suite}` 先验强度。\n")
        elif suite_m + 1e-12 < legacy:
            lines.append(f"\n> **{case['name']}**: suite_default 优于 legacy_full，数据集感知先验有效。\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
