"""
R15 诊断脚本：分析 4 个失败任务的精确瓶颈
对每个失败任务，逐步检查 R11-R16 的每个阶段
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/songjia9/Downloads/SRO-Autoresearch-autoresearch-may20')

from benchmarks.klv_benchmarks import KLV_BENCHMARKS
from train import ResidualSolver, build_var_data, N_GRID, eval_tree, is_valid_tree
from tool.tree_utils import tree_to_str
from tool.rational_mutation import deep_copy_tree


def diagnose_task(case):
    """诊断单个任务的每个阶段"""
    name = case["name"]
    profile = case["profile"]
    
    var_data = build_var_data(case, N_GRID)
    if case.get("dim", 1) == 2:
        y_obs = case["target"](var_data["x"], var_data["y"])
    else:
        y_obs = case["target"](var_data["x"])
    
    solver = ResidualSolver()
    solver.var_data = var_data
    solver.x_test = var_data.get("x", next(iter(var_data.values())))
    
    print(f"\n{'='*70}")
    print(f"诊断: {name} | 目标: {case['function_str']}")
    print(f"y_obs 范围: [{y_obs.min():.4f}, {y_obs.max():.4f}], std={y_obs.std():.4f}")
    print(f"{'='*70}")
    
    # 阶段 1: R13 模式库匹配
    print("\n[阶段 1] R13 模式库匹配...")
    from tool.pattern_library import get_pattern_library
    library = get_pattern_library()
    best_r13_mse = float('inf')
    best_r13_name = None
    for pattern in library.patterns:
        try:
            patch_y = eval_tree(pattern.tree, var_data)
            dot_prod = np.dot(patch_y, y_obs)
            norm_sq = np.dot(patch_y, patch_y)
            if norm_sq < 1e-12:
                continue
            w = dot_prod / norm_sq
            residual = y_obs - w * patch_y
            mse = float(np.mean(residual ** 2))
            if mse < best_r13_mse:
                best_r13_mse = mse
                best_r13_name = pattern.name
        except Exception:
            continue
    print(f"   最佳模式: {best_r13_name}, MSE={best_r13_mse:.6e}")
    
    # 阶段 2: R11 组合候选池构建
    print("\n[阶段 2] R11 组合候选池...")
    prior_trees = solver._build_prior_trees(profile)
    composed_trees = solver._build_composed_candidates(prior_trees, profile)
    print(f"   候选池大小: {len(composed_trees)}")
    
    # 检查目标分子结构是否存在
    target_numerators = {
        "Keijzer-4": None,  # 特殊结构
        "Livermore-7": "(x + y)",
        "Vladislavleva-4": "((x * x) * y)",
        "Vladislavleva-5": "(x * (y * y))",
    }
    target_denominators = {
        "Keijzer-4": None,
        "Livermore-7": "((x * y) + 1)",
        "Vladislavleva-4": "(x + y)",
        "Vladislavleva-5": "(x + y)",
    }
    
    target_num = target_numerators.get(name)
    target_den = target_denominators.get(name)
    
    if target_num:
        found_num = any(tree_to_str(t) == target_num for t in composed_trees if is_valid_tree(t))
        print(f"   目标分子 {target_num}: {'✓ 存在' if found_num else '✗ 不存在'}")
    if target_den:
        found_den = any(tree_to_str(t) == target_den for t in composed_trees if is_valid_tree(t))
        print(f"   目标分母 {target_den}: {'✓ 存在' if found_den else '✗ 不存在'}")
    
    # 阶段 3: 检查高价值种子
    print("\n[阶段 3] R11 高价值种子...")
    high_value_seeds = []
    for t in composed_trees:
        if not is_valid_tree(t):
            continue
        s = tree_to_str(t)
        if any(p in s for p in ['(x * x)', '(y * y)', '(x * y)', 'sqrt((x * y))']):
            high_value_seeds.append(t)
            if len(high_value_seeds) >= 50:
                break
    print(f"   高价值种子数: {len(high_value_seeds)}")
    
    # 列出包含目标变量的种子
    if target_num:
        matching_seeds = [t for t in high_value_seeds if tree_to_str(t) == target_num]
        print(f"   匹配目标分子的种子: {len(matching_seeds)}")
        if not matching_seeds:
            # 打印包含相关变量的种子
            for t in high_value_seeds[:10]:
                print(f"     - {tree_to_str(t)}")
    
    # 阶段 4: 检查加法结构
    print("\n[阶段 4] R11 加法结构...")
    additive_candidates = []
    for t in composed_trees:
        if not is_valid_tree(t):
            continue
        s = tree_to_str(t)
        if s in ['(x + y)', '(x - y)', '(y + x)', '(y - x)']:
            additive_candidates.append(t)
    print(f"   双变量加法候选: {len(additive_candidates)}")
    for t in additive_candidates[:5]:
        print(f"     - {tree_to_str(t)}")
    
    if target_den:
        found_den = any(tree_to_str(t) == target_den for t in additive_candidates)
        print(f"   目标分母 {target_den}: {'✓ 存在' if found_den else '✗ 不存在'}")
    
    # 阶段 5: 直接测试目标比值
    print("\n[阶段 5] 直接构造目标比值并评估...")
    from tool.Node import Node
    x = Node('x')
    y = Node('y')
    
    target_exprs = {
        "Keijzer-4": None,  # 特殊
        "Livermore-7": ("(x+y)", "(x*y+1)", lambda x, y: (x+y)/(x*y+1)),
        "Vladislavleva-4": ("x^2*y", "(x+y)", lambda x, y: x**2*y/(x+y)),
        "Vladislavleva-5": ("x*y^2", "(x+y)", lambda x, y: x*y**2/(x+y)),
    }
    
    if name in target_exprs and target_exprs[name] is not None:
        num_desc, den_desc, target_fn = target_exprs[name]
        
        # 构造对应的树
        target_trees = {
            "Livermore-7": (Node('+', x, y), Node('+', Node('*', x, y), Node('const', value=1.0))),
            "Vladislavleva-4": (Node('*', Node('*', x, x), y), Node('+', x, y)),
            "Vladislavleva-5": (Node('*', x, Node('*', y, y)), Node('+', x, y)),
        }
        
        if name in target_trees:
            num_tree, den_tree = target_trees[name]
            ratio_tree = Node('/', left=num_tree, right=den_tree)
            
            try:
                patch_y = eval_tree(ratio_tree, var_data)
                valid = not (np.any(np.isnan(patch_y)) or np.any(np.isinf(patch_y)))
                var_ok = np.var(patch_y) >= 1e-6
                
                print(f"   目标比值: {tree_to_str(ratio_tree)}")
                print(f"   数值有效: {valid}, 方差足够: {var_ok}")
                
                if valid and var_ok:
                    dot_prod = np.dot(patch_y, y_obs)
                    norm_sq = np.dot(patch_y, patch_y)
                    w = dot_prod / norm_sq
                    residual = y_obs - w * patch_y
                    mse = float(np.mean(residual ** 2))
                    print(f"   权重: {w:.6f}")
                    print(f"   MSE: {mse:.6e}")
                    
                    if abs(w - 1.0) < 1e-6:
                        print(f"   ✅ 精确匹配! (权重≈1)")
                    else:
                        print(f"   ⚠️ 权重偏离 1: {w:.6f}")
            except Exception as e:
                print(f"   ❌ 评估失败: {e}")
    
    # 阶段 6: 检查 R11 比值构造是否包含目标
    print("\n[阶段 6] R11 比值构造覆盖检查...")
    multiplicative_subset = (high_value_seeds + composed_trees)[:200]
    additive_subset_simple = [t for t in composed_trees if is_valid_tree(t) and 
                              tree_to_str(t) in ['(x + y)', '(x - y)', '(y + x)', '(y - x)']]
    
    # 检查是否能构造出目标比值
    target_ratio_strs = {
        "Livermore-7": None,  # 需要特殊分母
        "Vladislavleva-4": "(((x * x) * y) / (x + y))",
        "Vladislavleva-5": "((x * (y * y)) / (x + y))",
    }
    
    if name in target_ratio_strs and target_ratio_strs[name]:
        target_str = target_ratio_strs[name]
        found = False
        count = 0
        for ta in multiplicative_subset:
            for tb in additive_subset_simple:
                for num, den in [(ta, tb), (tb, ta)]:
                    ratio_tree = Node('/', left=deep_copy_tree(num), right=deep_copy_tree(den))
                    if is_valid_tree(ratio_tree):
                        s = tree_to_str(ratio_tree)
                        if s == target_str:
                            found = True
                        count += 1
                if count >= 10000:
                    break
            if count >= 10000:
                break
        
        print(f"   目标比值: {target_str}")
        print(f"   在构造空间中: {'✓ 存在' if found else '✗ 不存在'}")
        print(f"   总构造比值数: {count}")
    
    # 阶段 7: 分析 MSE 分布
    print("\n[阶段 7] R11 比值 MSE 分布...")
    all_mses = []
    for ta in multiplicative_subset[:50]:
        for tb in additive_subset_simple:
            for num, den in [(ta, tb), (tb, ta)]:
                ratio_tree = Node('/', left=deep_copy_tree(num), right=deep_copy_tree(den))
                if not is_valid_tree(ratio_tree):
                    continue
                try:
                    patch_y = eval_tree(ratio_tree, var_data)
                    if np.var(patch_y) < 1e-6 or np.any(np.isnan(patch_y)) or np.any(np.isinf(patch_y)):
                        continue
                    dot_prod = np.dot(patch_y, y_obs)
                    norm_sq = np.dot(patch_y, patch_y)
                    if norm_sq < 1e-12:
                        continue
                    w = dot_prod / norm_sq
                    residual = y_obs - w * patch_y
                    mse = float(np.mean(residual ** 2))
                    all_mses.append(mse)
                except Exception:
                    continue
    
    if all_mses:
        all_mses.sort()
        print(f"   评估比值数: {len(all_mses)}")
        print(f"   MSE min: {all_mses[0]:.6e}")
        print(f"   MSE 10%: {all_mses[len(all_mses)//10]:.6e}")
        print(f"   MSE 50%: {all_mses[len(all_mses)//2]:.6e}")
        print(f"   MSE 90%: {all_mses[len(all_mses)*9//10]:.6e}")
        print(f"   成功(<1e-4): {sum(1 for m in all_mses if m < 1e-4)}")
        print(f"   接近(<1e-2): {sum(1 for m in all_mses if m < 1e-2)}")


# 诊断 4 个失败任务
failed_cases = [c for c in KLV_BENCHMARKS if c["name"] in 
                ["Keijzer-4", "Livermore-7", "Vladislavleva-4", "Vladislavleva-5"]]

for case in failed_cases:
    diagnose_task(case)
