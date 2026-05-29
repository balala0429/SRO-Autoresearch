import argparse
import copy
import os
from datetime import datetime
from statistics import mean, pstdev

from benchmarks.extra_symbolic_benchmarks import EXTRA_BENCHMARKS
from benchmarks.unified_data_adapter import load_feynman_sr_tasks
from train import (
    NGUYEN_BENCHMARKS,
    PROFILE_CONFIG,
    PIPELINE_ABLATION_MODES,
    ResidualSolver,
    SUCCESS_MSE_THRESHOLD,
    build_var_data,
    resolve_suite,
    run_nguyen_repeated_benchmark,
)

# 五档流水线消融（论文主表）
PIPELINE_ABLATIONS = [
    ("no prior", "no_prior"),
    ("prior only", "prior_only"),
    ("retrieval only", "retrieval_only"),
    ("retrieval + splice", "retrieval_plus_splice"),
    ("full system", "full"),
]

# 旧版模块消融（--legacy）
MAIN_ABLATIONS = [
    ("Full Model", None),
    ("-Bootstrap Priors", "no_bootstrap"),
    ("-Nonlinear Splicing (raw only)", "raw_only"),
]

APPENDIX_ABLATIONS = [
    ("-Key Target Priors", "no_key_target"),
    ("-Integrity Filter", "no_integrity"),
    ("-Complexity Control", "weak_complexity"),
]


def _apply_legacy_ablation(kind):
    restore = []
    if kind is None:
        return restore
    if kind == "no_bootstrap":
        for cfg in PROFILE_CONFIG.values():
            cfg["bootstrap_priors"] = False
            cfg["bootstrap_mode"] = "off"
    elif kind == "raw_only":
        for cfg in PROFILE_CONFIG.values():
            cfg["splice_modes"] = ("raw",)
    elif kind == "no_key_target":
        old = ResidualSolver._build_prior_trees

        def patched(profile):
            priors = old(profile)
            if profile == "trig":
                priors = [p for p in priors if "sin((x * x)) * cos(x)" not in str(p.__dict__)]
            if profile == "log":
                priors = [p for p in priors if "log" not in getattr(p, "op", "")]
            if profile == "sqrt":
                priors = [p for p in priors if getattr(p, "op", "") != "sqrt"]
            return priors

        ResidualSolver._build_prior_trees = staticmethod(patched)
        restore.append(lambda: setattr(ResidualSolver, "_build_prior_trees", staticmethod(old)))
    elif kind == "no_integrity":
        old = ResidualSolver._filter_library
        ResidualSolver._filter_library = staticmethod(lambda index, trees: (index, trees))
        restore.append(lambda: setattr(ResidualSolver, "_filter_library", staticmethod(old)))
    elif kind == "weak_complexity":
        for cfg in PROFILE_CONFIG.values():
            cfg["penalty_weight"] = 0.0
            cfg["mse_tie_ratio"] = 0.0
            cfg["prune_weight"] = 1e-12
    return restore


def _stats(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def _eval_suite(solver, benchmarks, num_trials=30, suite=None, ablation_mode=None, return_details=False):
    per_task_mse = []
    details = []
    succ = 0
    for case in benchmarks:
        if "_y_obs" in case:
            var_data, y_obs = case["_var_data"], case["_y_obs"]
        else:
            var_data = build_var_data(case, 127)
            if case.get("dim", 1) == 2:
                y_obs = case["target"](var_data["x"], var_data["y"])
            else:
                y_obs = case["target"](var_data["x"])
        solver.var_data = var_data
        solver.x_test = var_data.get("x", next(iter(var_data.values())))
        task_suite = suite or resolve_suite(case)
        local = []
        for _ in range(num_trials):
            mse, _ = solver.solve(
                y_obs=y_obs,
                profile=case["profile"],
                suite=task_suite,
                ablation_mode=ablation_mode,
                plot=False,
                verbose=False,
            )
            local.append(float(mse))
        m = mean(local)
        per_task_mse.append(m)
        ok = m < SUCCESS_MSE_THRESHOLD
        if ok:
            succ += 1
        if return_details:
            details.append({"name": case["name"], "mse": m, "success": ok})
    if return_details:
        return per_task_mse, succ, len(benchmarks), details
    return per_task_mse, succ, len(benchmarks)


def _feynman_cases(max_count=8):
    tasks = load_feynman_sr_tasks(max_count=max_count)
    return [
        {
            "name": t.name,
            "dim": t.dim,
            "profile": t.profile,
            "suite": t.suite or "feynman",
            "source": t.source,
            "_var_data": t.var_data,
            "_y_obs": t.y_obs,
        }
        for t in tasks
    ]


def _run_pipeline_setting(tag, pipeline_mode, args):
    """pipeline_mode: no_prior | prior_only | retrieval_only | retrieval_plus_splice | full"""
    ablation_mode = None if pipeline_mode == "full" else pipeline_mode
    seed_rows = []
    for _seed in range(args.seeds):
        solver = ResidualSolver()
        avg_mse, n_success, n_eval, rows = run_nguyen_repeated_benchmark(
            solver,
            benchmarks=NGUYEN_BENCHMARKS,
            num_rounds=args.nguyen_rounds,
            num_trials=args.nguyen_trials,
            use_cache=False,
            verbose_per_case=False,
            ablation_mode=ablation_mode,
        )
        nguyen_mses = [float(r["Avg MSE"]) for r in rows]
        extra_mses, extra_succ, _, extra_det = _eval_suite(
            solver,
            EXTRA_BENCHMARKS,
            num_trials=args.extra_trials,
            suite="extra",
            ablation_mode=ablation_mode,
            return_details=True,
        )
        feyn_mses, feyn_succ, _, feyn_det = _eval_suite(
            solver,
            _feynman_cases(args.max_feynman),
            num_trials=args.feynman_trials,
            suite="feynman",
            ablation_mode=ablation_mode,
            return_details=True,
        )
        nguyen_det = [
            {
                "name": r["Dataset"],
                "mse": float(r["Avg MSE"]),
                "success": float(r["Avg MSE"]) < SUCCESS_MSE_THRESHOLD,
            }
            for r in rows
        ]
        seed_rows.append(
            {
                "nguyen_avg": mean(nguyen_mses),
                "nguyen_success": n_success,
                "extra_avg": mean(extra_mses),
                "extra_success": extra_succ,
                "feyn_avg": mean(feyn_mses),
                "feyn_success": feyn_succ,
                "nguyen_det": nguyen_det,
                "extra_det": extra_det,
                "feyn_det": feyn_det,
            }
        )

    def agg(key):
        return _stats([r[key] for r in seed_rows])

    ng_mean, ng_std = agg("nguyen_avg")
    ex_mean, ex_std = agg("extra_avg")
    fy_mean, fy_std = agg("feyn_avg")

    def _merge_task_details(det_key):
        names = set()
        for row in seed_rows:
            for d in row[det_key]:
                names.add(d["name"])
        merged = []
        for name in sorted(names):
            mses = []
            oks = 0
            for row in seed_rows:
                hit = next((d for d in row[det_key] if d["name"] == name), None)
                if hit:
                    mses.append(hit["mse"])
                    oks += int(hit["success"])
            merged.append(
                {
                    "name": name,
                    "mse_mean": mean(mses) if mses else float("inf"),
                    "success_rate": oks / max(1, len(seed_rows)),
                }
            )
        return merged

    return {
        "tag": tag,
        "pipeline_mode": pipeline_mode,
        "nguyen_mse_mean": ng_mean,
        "nguyen_mse_std": ng_std,
        "nguyen_success": f"{int(round(mean([r['nguyen_success'] for r in seed_rows])))}/{len(NGUYEN_BENCHMARKS)}",
        "extra_mse_mean": ex_mean,
        "extra_mse_std": ex_std,
        "extra_success": f"{int(round(mean([r['extra_success'] for r in seed_rows])))}/{len(EXTRA_BENCHMARKS)}",
        "feyn_mse_mean": fy_mean,
        "feyn_mse_std": fy_std,
        "feyn_success": f"{int(round(mean([r['feyn_success'] for r in seed_rows])))}/{args.max_feynman}",
        "seeds": args.seeds,
        "nguyen_tasks": _merge_task_details("nguyen_det"),
        "extra_tasks": _merge_task_details("extra_det"),
        "feyn_tasks": _merge_task_details("feyn_det"),
    }


def _run_legacy_setting(tag, ablation_kind, args):
    profile_backup = copy.deepcopy(PROFILE_CONFIG)
    restore_hooks = []
    seed_rows = []
    try:
        for _seed in range(args.seeds):
            solver = ResidualSolver()
            restore_hooks = _apply_legacy_ablation(ablation_kind)
            avg_mse, n_success, _n_eval, rows = run_nguyen_repeated_benchmark(
                solver,
                benchmarks=NGUYEN_BENCHMARKS,
                num_rounds=args.nguyen_rounds,
                num_trials=args.nguyen_trials,
                use_cache=False,
                verbose_per_case=False,
            )
            nguyen_mses = [float(r["Avg MSE"]) for r in rows]
            extra_mses, extra_succ, _ = _eval_suite(
                solver, EXTRA_BENCHMARKS, num_trials=args.extra_trials, suite="extra"
            )
            feyn_mses, feyn_succ, _ = _eval_suite(
                solver, _feynman_cases(args.max_feynman), num_trials=args.feynman_trials, suite="feynman"
            )
            seed_rows.append(
                {
                    "nguyen_avg": mean(nguyen_mses),
                    "nguyen_success": n_success,
                    "extra_avg": mean(extra_mses),
                    "extra_success": extra_succ,
                    "feyn_avg": mean(feyn_mses),
                    "feyn_success": feyn_succ,
                }
            )
            for fn in restore_hooks:
                fn()
            restore_hooks = []
            PROFILE_CONFIG.clear()
            PROFILE_CONFIG.update(copy.deepcopy(profile_backup))
    finally:
        PROFILE_CONFIG.clear()
        PROFILE_CONFIG.update(profile_backup)
        for fn in restore_hooks:
            fn()

    def agg(key):
        return _stats([r[key] for r in seed_rows])

    ng_mean, ng_std = agg("nguyen_avg")
    ex_mean, ex_std = agg("extra_avg")
    fy_mean, fy_std = agg("feyn_avg")
    return {
        "tag": tag,
        "nguyen_mse_mean": ng_mean,
        "nguyen_mse_std": ng_std,
        "nguyen_success": f"{int(round(mean([r['nguyen_success'] for r in seed_rows])))}/{len(NGUYEN_BENCHMARKS)}",
        "extra_mse_mean": ex_mean,
        "extra_mse_std": ex_std,
        "extra_success": f"{int(round(mean([r['extra_success'] for r in seed_rows])))}/{len(EXTRA_BENCHMARKS)}",
        "feyn_mse_mean": fy_mean,
        "feyn_mse_std": fy_std,
        "feyn_success": f"{int(round(mean([r['feyn_success'] for r in seed_rows])))}/{args.max_feynman}",
        "seeds": args.seeds,
    }


def _print_table(title, results):
    print(f"\n## {title}")
    print(
        "| Setting | Nguyen MSE (mean±std) | Nguyen Success | "
        "Extra MSE (mean±std) | Extra Success | "
        "Feynman MSE (mean±std) | Feynman Success |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['tag']} | {r['nguyen_mse_mean']:.3e}±{r['nguyen_mse_std']:.3e} | {r['nguyen_success']} | "
            f"{r['extra_mse_mean']:.3e}±{r['extra_mse_std']:.3e} | {r['extra_success']} | "
            f"{r['feyn_mse_mean']:.3e}±{r['feyn_mse_std']:.3e} | {r['feyn_success']} |"
        )


def _find_nguyen_case(name):
    for c in NGUYEN_BENCHMARKS:
        if c["name"].lower() == name.lower():
            return c
    return None


def _bootstrap_shortcut_check(case_name):
    """full 模式下是否 bootstrap 即收敛、未进入检索轮。"""
    case = _find_nguyen_case(case_name)
    if not case:
        return {"case": case_name, "error": "not found"}
    var_data = build_var_data(case, 127)
    if case.get("dim", 1) == 2:
        y_obs = case["target"](var_data["x"], var_data["y"])
    else:
        y_obs = case["target"](var_data["x"])
    solver = ResidualSolver()
    solver.var_data = var_data
    solver.x_test = var_data.get("x", next(iter(var_data.values())))
    mse, expr = solver.solve(
        y_obs=y_obs,
        profile=case["profile"],
        suite="nguyen",
        ablation_mode=None,
        verbose=False,
        trace=True,
    )
    trace = solver.solve_trace or []
    n_rounds = sum(1 for e in trace if e.get("event") == "round_pick")
    boot = next((e for e in trace if e.get("event") == "bootstrap"), None)
    return {
        "case": case_name,
        "mse": mse,
        "expr": expr,
        "bootstrap_only": n_rounds == 0 and boot is not None,
        "n_retrieval_rounds": n_rounds,
        "bootstrap_formula": boot.get("formula") if boot else None,
        "success": mse < SUCCESS_MSE_THRESHOLD,
    }


def _write_extended_report(path, results, protocol):
    full_r = next((r for r in results if r["pipeline_mode"] == "full"), None)
    rs_r = next((r for r in results if r["pipeline_mode"] == "retrieval_plus_splice"), None)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Pipeline Ablation Extended\n\n{protocol}\n\n")
        f.write("## Success ladder\n\n")
        f.write("| Setting | Nguyen | Extra | Feynman |\n|---|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r['tag']} | {r['nguyen_success']} | {r['extra_success']} | {r['feyn_success']} |\n"
            )
        if full_r and rs_r:
            f.write("\n## Full vs retrieval+splice (per task)\n\n")
            f.write("| Task | Suite | full succ | r+s succ | full MSE | r+s MSE | Δ |\n")
            f.write("|---|---|---:|---:|---:|---:|---:|\n")
            for suite_key, label in [
                ("nguyen_tasks", "nguyen"),
                ("extra_tasks", "extra"),
                ("feyn_tasks", "feynman"),
            ]:
                rs_map = {t["name"]: t for t in rs_r.get(suite_key, [])}
                for t in full_r.get(suite_key, []):
                    rs = rs_map.get(t["name"], {})
                    f_ok = t["success_rate"]
                    r_ok = rs.get("success_rate", 0.0)
                    f.write(
                        f"| {t['name']} | {label} | {f_ok:.0%} | {r_ok:.0%} | "
                        f"{t['mse_mean']:.3e} | {rs.get('mse_mean', float('nan')):.3e} | "
                        f"{f_ok - r_ok:+.0%} |\n"
                    )
            ng_f = int(full_r["nguyen_success"].split("/")[0])
            ng_r = int(rs_r["nguyen_success"].split("/")[0])
            ex_f = int(full_r["extra_success"].split("/")[0])
            ex_r = int(rs_r["extra_success"].split("/")[0])
            fy_f = int(full_r["feyn_success"].split("/")[0])
            fy_r = int(rs_r["feyn_success"].split("/")[0])
            f.write(
                f"\n**Gap summary:** Nguyen +{ng_f - ng_r}, Extra +{ex_f - ex_r}, Feynman +{fy_f - fy_r}\n"
            )
        f.write("\n## Bootstrap shortcut check (full mode)\n\n")
        for cname in ("Nguyen-5", "Nguyen-10", "Nguyen-6", "Nguyen-11"):
            info = _bootstrap_shortcut_check(cname)
            f.write(f"- **{cname}**: bootstrap_only={info.get('bootstrap_only')} ")
            f.write(f"retrieval_rounds={info.get('n_retrieval_rounds')} ")
            f.write(f"success={info.get('success')} mse={info.get('mse', 0):.3e}\n")
            if info.get("bootstrap_formula"):
                f.write(f"  - bootstrap formula: `{info['bootstrap_formula']}`\n")
            f.write(f"  - final expr: `{info.get('expr', '')[:120]}`\n")


def _write_report(path, title, protocol, results):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(protocol + "\n\n")
        f.write(
            "## Component ladder\n\n"
            "| Mode | Profile priors | Bootstrap | Prior in beam | FAISS | Splice |\n"
            "|------|:---:|:---:|:---:|:---:|:---:|\n"
            "| no prior | ✗ | ✗ | ✗ | ✗ | ✗ (var(y) baseline) |\n"
            "| prior only | ✓ | ✓ | ✗ | ✗ | raw |\n"
            "| retrieval only | ✗ | ✗ | ✗ | ✓ | raw |\n"
            "| retrieval + splice | ✗ | ✗ | ✗ | ✓ | profile |\n"
            "| full system | ✓ | ✓ | ✓ | ✓ | profile |\n\n"
        )
        f.write("## Results\n\n")
        f.write(
            "| Setting | Nguyen MSE | Extra MSE | Feynman MSE | "
            "Nguyen Succ | Extra Succ | Feynman Succ |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r['tag']} | {r['nguyen_mse_mean']:.3e}±{r['nguyen_mse_std']:.3e} | "
                f"{r['extra_mse_mean']:.3e}±{r['extra_mse_std']:.3e} | "
                f"{r['feyn_mse_mean']:.3e}±{r['feyn_mse_std']:.3e} | "
                f"{r['nguyen_success']} | {r['extra_success']} | {r['feyn_success']} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description="SRO ablation: pipeline ladder (default) or legacy module ablation")
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="五档流水线消融（默认开启，除非指定 --legacy）",
    )
    parser.add_argument("--legacy", action="store_true", help="旧版模块消融（Bootstrap/Splice/附录）")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--nguyen_rounds", type=int, default=1)
    parser.add_argument("--nguyen_trials", type=int, default=20)
    parser.add_argument("--extra_trials", type=int, default=30)
    parser.add_argument("--feynman_trials", type=int, default=30)
    parser.add_argument("--max_feynman", type=int, default=8)
    parser.add_argument("--main_only", action="store_true", help="仅 legacy 主文三项")
    parser.add_argument("--appendix_only", action="store_true", help="仅 legacy 附录")
    parser.add_argument("--quick", action="store_true", help="快速冒烟: seeds=1, trials=3")
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()

    if args.quick:
        args.seeds = 1
        args.nguyen_trials = 3
        args.extra_trials = 3
        args.feynman_trials = 3

    os.makedirs(args.out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    protocol = (
        f"Protocol: seeds={args.seeds}, Nguyen {args.nguyen_rounds}×{args.nguyen_trials} (no_cache), "
        f"Extra trials={args.extra_trials}, Feynman trials={args.feynman_trials}, "
        f"success MSE < {SUCCESS_MSE_THRESHOLD}"
    )

    run_pipeline = args.pipeline or not args.legacy
    if run_pipeline and not args.legacy:
        results = []
        for name, mode in PIPELINE_ABLATIONS:
            if mode not in PIPELINE_ABLATION_MODES:
                raise RuntimeError(f"unknown mode {mode}")
            print("\n" + "=" * 60)
            print(f"[PIPELINE] {name} ({mode})")
            print("=" * 60)
            results.append(_run_pipeline_setting(name, mode, args))
        _print_table("Pipeline Ablation", results)
        print("\n## Success ladder (Nguyen / Extra / Feynman counts)")
        for r in results:
            ng = int(r["nguyen_success"].split("/")[0])
            ex = int(r["extra_success"].split("/")[0])
            fy = int(r["feyn_success"].split("/")[0])
            print(f"  {r['tag']:22s}  Nguyen={ng:2d}  Extra={ex}  Feynman={fy}")
        out_path = os.path.join(args.out_dir, f"ablation_pipeline_{tag}.md")
        ext_path = os.path.join(args.out_dir, f"ablation_pipeline_{tag}_extended.md")
        _write_report(out_path, f"Pipeline Ablation ({tag})", protocol, results)
        _write_extended_report(ext_path, results, protocol)
        print(f"\nSaved: {out_path}")
        print(f"Saved: {ext_path}")
        if full_r := next((r for r in results if r["pipeline_mode"] == "full"), None):
            if rs_r := next((r for r in results if r["pipeline_mode"] == "retrieval_plus_splice"), None):
                print(
                    f"\nFull vs r+s gap: Nguyen +{int(full_r['nguyen_success'].split('/')[0]) - int(rs_r['nguyen_success'].split('/')[0])}, "
                    f"Extra +{int(full_r['extra_success'].split('/')[0]) - int(rs_r['extra_success'].split('/')[0])}, "
                    f"Feynman +{int(full_r['feyn_success'].split('/')[0]) - int(rs_r['feyn_success'].split('/')[0])}"
                )
        return

    main_results = []
    appendix_results = []
    if not args.appendix_only:
        for name, kind in MAIN_ABLATIONS:
            print("\n" + "=" * 60)
            print(f"[LEGACY] {name}")
            print("=" * 60)
            main_results.append(_run_legacy_setting(name, kind, args))
    if not args.main_only:
        for name, kind in APPENDIX_ABLATIONS:
            print("\n" + "=" * 60)
            print(f"[LEGACY APPENDIX] {name}")
            print("=" * 60)
            appendix_results.append(_run_legacy_setting(name, kind, args))
    _print_table("Legacy Main", main_results)
    if appendix_results:
        _print_table("Legacy Appendix", appendix_results)
    out_path = os.path.join(args.out_dir, f"ablation_multisuite_{tag}.md")
    all_rows = main_results + appendix_results
    _write_report(out_path, f"Legacy Ablation ({tag})", protocol, all_rows)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
