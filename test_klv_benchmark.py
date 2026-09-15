"""
KLV Benchmark 完整测试脚本
测试 32 个 KLV 任务的精度，记录每个任务的成功率和 MSE
"""

import sys
import os
import json
from datetime import datetime

import numpy as np

sys.path.insert(0, '/Users/songjia9/Downloads/SRO-Autoresearch-autoresearch-may20')

from benchmarks.klv_benchmarks import KLV_BENCHMARKS
from train import ResidualSolver, build_var_data, N_GRID, SUCCESS_MSE_THRESHOLD


def run_klv_benchmark(num_rounds: int = 10, num_trials: int = 10, verbose: bool = True):
    """
    运行 KLV benchmark 测试
    
    Args:
        num_rounds: 测试轮数
        num_trials: 每轮试验次数
        verbose: 是否打印详细信息
    
    Returns:
        results: 每个任务的结果列表
    """
    total_trials = num_rounds * num_trials
    results = []
    
    solver = ResidualSolver()
    
    print("\n" + "=" * 80)
    print(f"KLV Benchmark Test | {len(KLV_BENCHMARKS)} tasks | {num_rounds} rounds x {num_trials} trials")
    print("=" * 80)
    
    for i, case in enumerate(KLV_BENCHMARKS):
        name = case["name"]
        profile = case["profile"]
        
        # 构建数据
        var_data = build_var_data(case, N_GRID)
        if case.get("dim", 1) == 2:
            y_obs = case["target"](var_data["x"], var_data["y"])
        else:
            y_obs = case["target"](var_data["x"])
        
        solver.var_data = var_data
        solver.x_test = var_data.get("x", next(iter(var_data.values())))
        
        success_count = 0
        mse_sum = 0.0
        best_mse = float("inf")
        best_expr = "0"
        
        if verbose:
            print(f"\n[{i+1:2d}/32] {name}: {case['function_str']}")
        
        for trial in range(total_trials):
            try:
                mse, expr = solver.solve(
                    y_obs=y_obs,
                    profile=profile,
                    suite="klv",
                    plot=False,
                    verbose=False,
                )
                mse_sum += mse
                if mse < best_mse:
                    best_mse = mse
                    best_expr = expr
                if mse < SUCCESS_MSE_THRESHOLD:
                    success_count += 1
            except Exception as e:
                if verbose:
                    print(f"   Trial {trial+1} failed: {e}")
                mse_sum += float("inf")
        
        avg_mse = mse_sum / total_trials
        success_rate = success_count / total_trials
        
        result = {
            "name": name,
            "function_str": case["function_str"],
            "profile": profile,
            "success_count": success_count,
            "total_trials": total_trials,
            "success_rate": success_rate,
            "avg_mse": float(avg_mse),
            "best_mse": float(best_mse),
            "best_expr": best_expr,
        }
        results.append(result)
        
        if verbose:
            status = "✓" if success_count == total_trials else "✗"
            print(f"   {status} Success: {success_count}/{total_trials} ({success_rate*100:.1f}%)")
            print(f"   Best MSE: {best_mse:.6e}")
            print(f"   Avg MSE: {avg_mse:.6e}")
            print(f"   Expression: {best_expr}")
    
    # 统计汇总
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    keijzer_results = [r for r in results if r["name"].startswith("Keijzer")]
    livermore_results = [r for r in results if r["name"].startswith("Livermore")]
    vladislavleva_results = [r for r in results if r["name"].startswith("Vladislavleva")]
    
    def count_exact(results):
        """精确恢复（全部成功）"""
        return sum(1 for r in results if r["success_count"] == r["total_trials"])
    
    k_exact = count_exact(keijzer_results)
    l_exact = count_exact(livermore_results)
    v_exact = count_exact(vladislavleva_results)
    total_exact = k_exact + l_exact + v_exact
    
    print(f"Keijzer:       {k_exact:2d}/13 ({k_exact/13*100:.1f}%)")
    print(f"Livermore:     {l_exact:2d}/12 ({l_exact/12*100:.1f}%)")
    print(f"Vladislavleva: {v_exact:2d}/7  ({v_exact/7*100:.1f}%)")
    print(f"Total:         {total_exact:2d}/32 ({total_exact/32*100:.1f}%)")
    
    # 部分成功的任务
    partial = [r for r in results if 0 < r["success_count"] < r["total_trials"]]
    if partial:
        print(f"\n部分成功的任务 ({len(partial)}):")
        for r in partial:
            print(f"  {r['name']}: {r['success_count']}/{r['total_trials']} ({r['success_rate']*100:.1f}%)")
    
    # 完全失败的任务
    failed = [r for r in results if r["success_count"] == 0]
    if failed:
        print(f"\n完全失败的任务 ({len(failed)}):")
        for r in failed:
            print(f"  {r['name']}: best_mse={r['best_mse']:.2e}")
    
    return results


def save_results(results, output_dir: str = "reports"):
    """保存结果到 JSON 文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"klv_benchmark_{timestamp}.json")
    
    # 计算汇总统计
    keijzer = [r for r in results if r["name"].startswith("Keijzer")]
    livermore = [r for r in results if r["name"].startswith("Livermore")]
    vladislavleva = [r for r in results if r["name"].startswith("Vladislavleva")]
    
    summary = {
        "timestamp": timestamp,
        "total_tasks": len(results),
        "keijzer_success": sum(1 for r in keijzer if r["success_count"] == r["total_trials"]),
        "livermore_success": sum(1 for r in livermore if r["success_count"] == r["total_trials"]),
        "vladislavleva_success": sum(1 for r in vladislavleva if r["success_count"] == r["total_trials"]),
        "total_success": sum(1 for r in results if r["success_count"] == r["total_trials"]),
    }
    summary["success_rate"] = summary["total_success"] / summary["total_tasks"]
    
    output = {
        "summary": summary,
        "results": results,
    }
    
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存: {json_path}")
    return json_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10, help="测试轮数")
    parser.add_argument("--trials", type=int, default=10, help="每轮试验次数")
    parser.add_argument("--output", type=str, default="reports", help="输出目录")
    parser.add_argument("--verbose", action="store_true", default=True, help="详细输出")
    args = parser.parse_args()
    
    results = run_klv_benchmark(
        num_rounds=args.rounds,
        num_trials=args.trials,
        verbose=args.verbose,
    )
    
    save_results(results, args.output)
