#!/usr/bin/env python3
"""单题逐步 trace：确认公式是逐轮构建，而非 accidental fit。"""
import argparse
import json
import os
from datetime import datetime

from benchmarks.extra_symbolic_benchmarks import EXTRA_BENCHMARKS
from benchmarks.unified_data_adapter import load_feynman_sr_tasks
from train import (
    NGUYEN_BENCHMARKS,
    N_GRID,
    SUCCESS_MSE_THRESHOLD,
    ResidualSolver,
    build_var_data,
    resolve_suite,
)


def _find_case(name: str):
    name_l = name.lower().replace("_", "-")
    for case in NGUYEN_BENCHMARKS + EXTRA_BENCHMARKS:
        if case["name"].lower() == name_l or case["name"].lower().replace("-", "") == name_l.replace("-", ""):
            return case
    for t in load_feynman_sr_tasks(max_count=50):
        if t.name.lower() == name_l:
            return {
                "name": t.name,
                "function_str": t.function_str,
                "profile": t.profile,
                "suite": t.suite or "feynman",
                "dim": t.dim,
                "_var_data": t.var_data,
                "_y_obs": t.y_obs,
            }
    raise SystemExit(f"未找到数据集: {name}（支持 Nguyen-* / Extra / Feynman）")


def _load_obs(case):
    if "_y_obs" in case:
        return case["_var_data"], case["_y_obs"]
    var_data = build_var_data(case, N_GRID)
    if case.get("dim", 1) == 2:
        y_obs = case["target"](var_data["x"], var_data["y"])
    else:
        y_obs = case["target"](var_data["x"])
    return var_data, y_obs


def _print_trace(trace, verbose):
    if not trace:
        print("(无 trace 记录)")
        return
    for ev in trace:
        et = ev.get("event")
        if et == "start":
            print(
                f"\n=== START | profile={ev['profile']} suite={ev['suite']} "
                f"mode={ev['ablation_mode']} prior_atoms={ev['n_prior_atoms']} ==="
            )
        elif et == "bootstrap":
            print(f"\n[bootstrap] {ev['n_terms']} terms | MSE={ev['mse']:.6e}")
            for t in ev.get("terms", []):
                print(f"  + {t}")
            print(f"  formula: {ev.get('formula', '')}")
        elif et == "round_start":
            print(f"\n--- round {ev['round']} | residual_mse={ev['residual_mse']:.6e} | active={ev['n_active']} ---")
        elif et == "round_pick":
            print(f"\n>>> round {ev['round']} PICK")
            print(f"    candidate = {ev['candidate']}")
            print(f"    splice      = {ev['splice']}")
            print(f"    added       = {ev.get('added_terms', [])}")
            print(f"    pick_mse    = {ev['pick_mse']:.6e}")
            print(f"    cumulative  = {ev['cumulative_mse']:.6e}")
            print(f"    formula     = {ev['cumulative_formula']}")
            if verbose:
                print("    [candidates evaluated]")
                for c in sorted(ev.get("candidates", []), key=lambda x: x["mse"])[:12]:
                    spl = c.get("best_splice", "raw")
                    print(
                        f"      {c['source']:5s} sim={c['similarity']:.3f} "
                        f"candidate={c['candidate'][:50]} splice={spl} mse={c['mse']:.6e}"
                    )
                    if verbose and c.get("variants"):
                        for v in c["variants"]:
                            print(f"        variant {v['splice']:10s} mse={v['mse']:.6e} terms={v['terms']}")
        elif et == "round_no_pick":
            print(f"\n>>> round {ev['round']} NO PICK (停止迭代)")
        elif et == "converged":
            print(f"\n[converged] round={ev['round']} mse={ev['mse']:.6e}")
        elif et == "finish":
            print(f"\n=== FINISH | final_mse={ev['final_mse']:.6e} terms={ev['n_terms']} ===")
            print(f"F = {ev['formula']}")


def main():
    parser = argparse.ArgumentParser(description="单题 SRO trace（逐步公式构建）")
    parser.add_argument("--dataset", type=str, required=True, help="如 Nguyen-5")
    parser.add_argument("--verbose", action="store_true", help="打印每轮全部候选变体")
    parser.add_argument("--suite", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "no_prior", "prior_only", "retrieval_only", "retrieval_plus_splice"],
    )
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()

    case = _find_case(args.dataset)
    var_data, y_obs = _load_obs(case)
    suite = args.suite or resolve_suite(case)
    profile = case["profile"]

    print(f"Dataset : {case['name']}")
    print(f"Target  : {case.get('function_str', '')}")
    print(f"Profile : {profile} | Suite: {suite} | Mode: {args.mode}")
    print(f"Threshold: MSE < {SUCCESS_MSE_THRESHOLD}")

    solver = ResidualSolver()
    solver.var_data = var_data
    solver.x_test = var_data.get("x", next(iter(var_data.values())))

    ablation_mode = None if args.mode == "full" else args.mode
    mse, expr = solver.solve(
        y_obs=y_obs,
        profile=profile,
        suite=suite,
        ablation_mode=ablation_mode,
        verbose=args.verbose,
        trace=True,
    )

    _print_trace(solver.solve_trace, verbose=args.verbose)

    ok = mse < SUCCESS_MSE_THRESHOLD
    print(f"\nResult: {'SUCCESS' if ok else 'FAIL'} | MSE={mse:.6e}")
    print(f"Expr  : {expr}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = case["name"].replace("/", "-")
    out_path = os.path.join(args.out_dir, f"trace_{safe}_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": case["name"],
                "function": case.get("function_str", ""),
                "mode": args.mode,
                "final_mse": mse,
                "final_expr": expr,
                "success": ok,
                "trace": solver.solve_trace,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved trace JSON: {out_path}")


if __name__ == "__main__":
    main()
