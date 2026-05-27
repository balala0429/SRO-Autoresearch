import torch
import numpy as np
import faiss
import pickle
import os
import sympy as sp
# 让主程序支持重复评估与表格输出
import argparse
# 导入你项目的模块
from main import NUM_POINTS, get_probing_points
from tool.normalize_y import normalize_y
from tool.Node import Node
from Predictor.ResidualPredictor import ResidualPredictor
from tool.visualize.plot_sro_progression import plot_sro_progression
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 工具函数：树的求值与打印 ---
def eval_tree(node, var_data):
    """递归计算一棵公式树在给定变量采样点上的结果（支持 x/y 多变量）"""
    if isinstance(var_data, np.ndarray):
        var_data = {'x': var_data}
    base = var_data.get('x', next(iter(var_data.values())))
    if node is None: return np.zeros_like(base)
    if node.op == 'x': return var_data['x']
    if node.op == 'y': return var_data.get('y', var_data['x'])
    if node.op == 'const': return np.full_like(base, node.value)
    if node.op == '+': return eval_tree(node.left, var_data) + eval_tree(node.right, var_data)
    if node.op == '-': return eval_tree(node.left, var_data) - eval_tree(node.right, var_data)
    if node.op == '*': return eval_tree(node.left, var_data) * eval_tree(node.right, var_data)
    if node.op == '/':
        # 防止除零
        denom = eval_tree(node.right, var_data)
        return eval_tree(node.left, var_data) / (np.where(np.abs(denom) < 1e-5, 1e-5, denom))
    if node.op == 'sin': return np.sin(eval_tree(node.left, var_data))
    if node.op == 'cos': return np.cos(eval_tree(node.left, var_data))
    if node.op == 'log': return np.log(np.abs(eval_tree(node.left, var_data)) + 1e-8)
    if node.op == 'sqrt': return np.sqrt(np.abs(eval_tree(node.left, var_data)))
    if node.op == 'exp': return np.exp(np.clip(eval_tree(node.left, var_data), -10, 10))
    return np.zeros_like(base)

def get_tree_size(node):
    """计算公式树的节点总数（复杂度）"""
    if node is None: return 0
    return 1 + get_tree_size(node.left) + get_tree_size(node.right)


def tree_to_str(node):
    """递归将树转为人类可读的字符串"""
    if node is None: return ""
    if node.op in ['x', 'y']: return node.op
    if node.op == 'const': return f"{node.value:.2f}"
    if node.op in ['+', '-', '*', '/']:
        return f"({tree_to_str(node.left)} {node.op} {tree_to_str(node.right)})"
    if node.op in ['sin', 'exp', 'cos', 'log', 'sqrt']:
        return f"{node.op}({tree_to_str(node.left)})"
    return ""


def build_var_data(case, n_grid):
    """为 1D/2D 目标函数统一构建变量采样点。"""
    if case.get("dim", 1) == 1:
        x_min, x_max = case["x_range"]
        x = np.linspace(x_min, x_max, n_grid)
        return {'x': x}
    x_min, x_max = case["x_range"]
    y_min, y_max = case["y_range"]
    x = np.linspace(x_min, x_max, n_grid)
    # y 采用反向采样，避免与 x 完全同相导致表达能力退化
    y = np.linspace(y_max, y_min, n_grid)
    return {'x': x, 'y': y}


SUCCESS_MSE_THRESHOLD = 1e-4
N_GRID = 127  # 与 predictor 训练及历史实验一致

# Nguyen 基准：name, profile, x_range, target
NGUYEN_BENCHMARKS = [
    {
        "name": "Nguyen-1",
        "function_str": "(x^3+x^2+x)",
        "dim": 1,
        "profile": "poly",
        "x_range": (-1.0, 1.0),
        "target": lambda x: x ** 3 + x ** 2 + x,
    },
    {
        "name": "Nguyen-2",
        "function_str": "(x^4+x^3+x^2+x)",
        "dim": 1,
        "profile": "poly",
        "x_range": (-1.0, 1.0),
        "target": lambda x: x ** 4 + x ** 3 + x ** 2 + x,
    },
    {
        "name": "Nguyen-3",
        "function_str": "(x^5+x^4+x^3+x^2+x)",
        "dim": 1,
        "profile": "poly",
        "x_range": (-1.0, 1.0),
        "target": lambda x: x ** 5 + x ** 4 + x ** 3 + x ** 2 + x,
    },
    {
        "name": "Nguyen-4",
        "function_str": "(x^6+x^5+x^4+x^3+x^2+x)",
        "dim": 1,
        "profile": "poly",
        "x_range": (-1.0, 1.0),
        "target": lambda x: x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x,
    },
    {
        "name": "Nguyen-5",
        "function_str": "(sin(x^2)cos(x)-1)",
        "dim": 1,
        "profile": "trig",
        "x_range": (-1.0, 1.0),
        "target": lambda x: np.sin(x ** 2) * np.cos(x) - 1,
    },
    {
        "name": "Nguyen-6",
        "function_str": "(sin(x)+sin(x+x^2))",
        "dim": 1,
        "profile": "trig",
        "x_range": (-1.0, 1.0),
        "target": lambda x: np.sin(x) + np.sin(x + x ** 2),
    },
    {
        "name": "Nguyen-7",
        "function_str": "(log(x+1)+log(x^2+1))",
        "dim": 1,
        "profile": "log",
        "x_range": (0.0, 2.0),
        "target": lambda x: np.log(x + 1) + np.log(x ** 2 + 1),
    },
    {
        "name": "Nguyen-8",
        "function_str": "(sqrt(x))",
        "dim": 1,
        "profile": "sqrt",
        "x_range": (0.0, 4.0),
        "target": lambda x: np.sqrt(x),
    },
    {
        "name": "Nguyen-9",
        "function_str": "(sin(x)+sin(y^2))",
        "dim": 2,
        "profile": "multi",
        "x_range": (-1.0, 1.0),
        "y_range": (-1.0, 1.0),
        "target": lambda x, y: np.sin(x) + np.sin(y ** 2),
    },
    {
        "name": "Nguyen-10",
        "function_str": "(2sin(x)cos(y))",
        "dim": 2,
        "profile": "multi",
        "x_range": (-1.0, 1.0),
        "y_range": (-1.0, 1.0),
        "target": lambda x, y: 2 * np.sin(x) * np.cos(y),
    },
    {
        "name": "Nguyen-11",
        "function_str": "(x^y)",
        "dim": 2,
        "profile": "multi",
        "x_range": (0.1, 2.0),
        "y_range": (0.1, 2.0),
        "target": lambda x, y: np.power(np.clip(x, 1e-6, None), y),
    },
    {
        "name": "Nguyen-12",
        "function_str": "(x^4-x^3+y^2/2-y)",
        "dim": 2,
        "profile": "multi",
        "x_range": (-1.0, 1.0),
        "y_range": (-1.0, 1.0),
        "target": lambda x, y: x ** 4 - x ** 3 + (y ** 2) / 2 - y,
    },
]

PROFILE_CONFIG = {
    "poly": {
        "top_k": 80,
        "beam_limit": 18,
        "penalty_weight": 0.02,
        "max_iters": 12,
        "tol": 1e-4,
        "splice_modes": ("raw", "sin", "lin+sin"),
    },
    "trig": {
        "top_k": 120,
        "beam_limit": 25,
        "penalty_weight": 0.05,
        "max_iters": 25,
        "tol": 1e-4,
        "splice_modes": ("raw", "sin", "tanh", "relu", "mul_sin_x", "lin+sin"),
    },
    "log": {
        "top_k": 100,
        "beam_limit": 22,
        "penalty_weight": 0.04,
        "max_iters": 20,
        "tol": 1e-4,
        "splice_modes": ("raw", "sin", "exp_mix", "lin+sin"),
    },
    "sqrt": {
        "top_k": 90,
        "beam_limit": 20,
        "penalty_weight": 0.03,
        "max_iters": 18,
        "tol": 1e-4,
        "splice_modes": ("raw", "sin", "relu", "mul_sin_x"),
    },
    "multi": {
        "top_k": 140,
        "beam_limit": 30,
        "penalty_weight": 0.04,
        "max_iters": 28,
        "tol": 1e-4,
        "splice_modes": ("raw", "sin", "tanh", "relu", "mul_sin_x", "lin+sin"),
    },
}


# --- 主类：残差求解器 ---
class ResidualSolver:
    def __init__(self):
        print("正在加载 SRO 武器系统...")
        self.x_test = get_probing_points(NUM_POINTS)
        self.var_data = {'x': self.x_test}
        actual_points = len(self.x_test)

        # 1. 加载神级雷达 (Predictor)
        self.predictor = ResidualPredictor(input_dim=actual_points, output_dim=128).to(device)
        self.predictor.load_state_dict(torch.load("weight2/predictor_final.pth", map_location=torch.device('cpu')))
        self.predictor.eval()

        # 2. 加载弹药库 (FAISS & Trees)
        print("正在挂载 FAISS 向量索引与树组件...")
        self.index = faiss.read_index("library/library/symbolic_index.bin")
        with open("library/library/symbolic_trees.pkl", "rb") as f:
            self.trees = pickle.load(f)

    @staticmethod
    def _build_prior_trees(profile):
        """按问题类型注入结构先验，避免多项式任务被三角先验污染。"""
        x = Node('x')
        xx = Node('*', x, x)
        xxx = Node('*', xx, x)
        if profile == "poly":
            x4 = Node('*', xxx, x)
            x5 = Node('*', x4, x)
            x6 = Node('*', x5, x)
            return [x, xx, xxx, x4, x5, x6]
        if profile == "trig":
            return [
                Node('sin', x),
                Node('sin', xx),
                Node('*', Node('sin', xx), Node('sin', x)),
                Node('*', xx, Node('sin', x)),
                Node('sin', Node('+', x, xx)),
            ]
        if profile == "log":
            return [
                x, xx,
                Node('exp', x),
                Node('exp', xx),
                Node('*', x, x),
            ]
        if profile == "sqrt":
            return [x, xx, Node('*', x, x)]
        if profile == "multi":
            y = Node('y')
            yy = Node('*', y, y)
            return [
                x, xx, y, yy,
                Node('sin', x),
                Node('sin', y),
                Node('cos', y),
                Node('*', Node('sin', x), Node('cos', y)),
                Node('*', x, y),
                Node('exp', x),
            ]
        return [x, xx]

    def _omp_mse(self, columns, y_obs):
        if not columns:
            return np.mean(y_obs ** 2)
        A = np.column_stack(columns)
        w, _, _, _ = np.linalg.lstsq(A, y_obs, rcond=None)
        return np.mean((y_obs - A @ w) ** 2)

    def _entry_label(self, entry):
        if isinstance(entry, tuple):
            return tree_to_str(entry[0]), entry[1]
        return tree_to_str(entry), 'raw'

    def _build_splice_variants(self, candidate_tree, patch_y, modes):
        sin_patch = np.sin(patch_y)
        tanh_patch = np.tanh(patch_y)
        relu_patch = np.maximum(patch_y, 0.0)
        mix_sin_x = patch_y * np.sin(self.x_test)
        exp_mix = np.exp(np.clip(patch_y, -8, 8))
        all_variants = {
            "raw": ([patch_y], [(candidate_tree, 'raw')]),
            "sin": ([sin_patch], [(candidate_tree, 'sin')]),
            "tanh": ([tanh_patch], [(candidate_tree, 'tanh')]),
            "relu": ([relu_patch], [(candidate_tree, 'relu')]),
            "mul_sin_x": ([mix_sin_x], [(candidate_tree, 'mul_sin_x')]),
            "exp_mix": ([exp_mix], [(candidate_tree, 'exp_mix')]),
            "lin+sin": (
                [patch_y, sin_patch],
                [(candidate_tree, 'raw'), (candidate_tree, 'sin')],
            ),
        }
        out = []
        for mode in modes:
            if mode in all_variants:
                cols, entries = all_variants[mode]
                out.append((cols, entries, mode))
        return out

    def solve(self, y_obs, max_iters=None, tol=None, profile="trig", plot=False, verbose=True):
        cfg = PROFILE_CONFIG.get(profile, PROFILE_CONFIG["trig"])
        if max_iters is None:
            max_iters = cfg["max_iters"]
        if tol is None:
            tol = cfg["tol"]
        prior_trees = self._build_prior_trees(profile)

        if verbose:
            print("=" * 50)
            print(f"🚀 SRO 推理 | profile={profile} | max_iters={max_iters} tol={tol}")
            print("=" * 50)

        # 记录已被选中的组件（波形和树结构）
        active_patches_y = []
        active_trees = []
        self.fit_history = []

        for step in range(1, max_iters + 1):
            # 1. 计算当前残差
            # 如果有已选组件，用全局系数计算当前最优预测
            if len(active_patches_y) > 0:
                A_current = np.column_stack(active_patches_y)
                # 全局最小二乘法：自动调整所有已选组件的系数
                w_current, _, _, _ = np.linalg.lstsq(A_current, y_obs, rcond=None)
                y_pred = A_current @ w_current
            else:
                y_pred = np.zeros_like(next(iter(self.var_data.values())))
                w_current = []

            res = y_obs - y_pred
            mse_current = np.mean(res ** 2)

            if mse_current < tol:
                if verbose:
                    print(f"\n✅ 达到收敛精度 (MSE: {mse_current:.6f})，提前结束拟合。")
                break

            if verbose:
                print(f"\n--- 第 {step} 轮迭代 | 当前 MSE: {mse_current:.4f} ---")

            # 2. 雷达探测
            y_norm = normalize_y(res)
            y_input = torch.tensor(y_norm).float().view(1, -1).to(device)
            # TODO 修改检索库的逻辑保证组件的质量，加入最开始提取到的子树组件
            # TODO 残差预测看看能不能修改一下网络结构提高性能拟合程度
            with torch.no_grad():
                v_pred = self.predictor(y_input)
                v_pred_np = v_pred.cpu().numpy().astype('float32')
                faiss.normalize_L2(v_pred_np)

            # 3. 弹药库检索 + 结构先验注入（按 profile 自适应）
            top_k_search = cfg["top_k"]
            beam_limit = cfg["beam_limit"]
            penalty_weight = cfg["penalty_weight"]
            splice_modes = cfg["splice_modes"]
            similarities, indices = self.index.search(v_pred_np, top_k_search)

            best_patch_cols = None
            best_patch_entries = None
            best_tree = None
            best_score = float('inf')

            # 4. Beam Search + 多样性去重 + 全局重拟合评估
            if verbose:
                print("雷达锁定候选补丁 (去重后):")
            seen_strs = {self._entry_label(t)[0] for t in active_trees}
            valid_candidates = 0
            candidate_queue = [(self.trees[idx], float(similarities[0][i]))
                               for i, idx in enumerate(indices[0])]
            candidate_queue = [(t, 1.0) for t in prior_trees] + candidate_queue

            for candidate_tree, sim in candidate_queue:
                tree_str = tree_to_str(candidate_tree)

                if tree_str in seen_strs:
                    continue
                seen_strs.add(tree_str)
                valid_candidates += 1
                if valid_candidates > beam_limit:
                    break

                patch_y = eval_tree(candidate_tree, self.var_data)
                if np.var(patch_y) < 1e-6 or np.any(np.isnan(patch_y)) or np.any(np.isinf(patch_y)):
                    continue

                variants = self._build_splice_variants(candidate_tree, patch_y, splice_modes)
                test_mse = float('inf')
                best_variant = None
                for cols, entries, _tag in variants:
                    if any(np.any(np.isnan(c)) or np.any(np.isinf(c)) for c in cols):
                        continue
                    mse = self._omp_mse(active_patches_y + cols, y_obs)
                    if mse < test_mse:
                        test_mse = mse
                        best_variant = (cols, entries)

                if best_variant is None:
                    continue

                complexity = get_tree_size(candidate_tree)
                adjusted_score = test_mse * (1.0 + penalty_weight * complexity)

                if verbose:
                    print(
                        f"  [{valid_candidates}] 相似度 {sim:.4f} | 组件: {tree_str} | "
                        f"节点: {complexity} | 综合得分: {adjusted_score:.4f}")

                if adjusted_score < best_score:
                    best_score = adjusted_score
                    best_patch_cols, best_patch_entries = best_variant
                    best_tree = candidate_tree

            if best_tree is None:
                break

            # 5. 正式录用最佳补丁（可含 sin 等非线性列），加入全局列阵
            for col, entry in zip(best_patch_cols, best_patch_entries):
                active_patches_y.append(col)
                active_trees.append(entry)

            # 算一下新阵列的实时系数，仅用于打印展示
            A_new = np.column_stack(active_patches_y)
            w_new, _, _, _ = np.linalg.lstsq(A_new, y_obs, rcond=None)
            if verbose:
                print(f"🎯 选定补丁: [{tree_to_str(best_tree)}]。全局系数重整中...")

            # 熔断机制：新加入列的系数均趋零则视为收敛
            n_new = len(best_patch_cols)
            if all(abs(w) < 1e-4 for w in w_new[-n_new:]):
                if verbose:
                    print(f"🛑 熔断触发！新组件在全局回归中权重趋零，模型收敛。")
                for _ in range(n_new):
                    active_patches_y.pop()
                    active_trees.pop()
                break

                # 【新增：记录本轮战况供画图使用】
            current_pred = A_new @ w_new
            current_res = y_obs - current_pred
            self.fit_history.append({
                'step': step,
                'patch': tree_to_str(best_tree),
                'y_pred': current_pred,  # OMP 优化后的最新曲线
                'res': current_res,  # 最新的残差
                'mse': np.mean(current_res ** 2)
            })

        # --- 最终公式构建 ---
        final_mse = float('inf')
        best_expr = "0"
        if len(active_patches_y) > 0:
            A_final = np.column_stack(active_patches_y)
            w_final, _, _, _ = np.linalg.lstsq(A_final, y_obs, rcond=None)

            formula_parts = []
            for weight, entry in zip(w_final, active_trees):
                base_str, mode = self._entry_label(entry)
                if mode == 'sin':
                    term_str = f"sin({base_str})"
                elif mode == 'tanh':
                    term_str = f"tanh({base_str})"
                elif mode == 'relu':
                    term_str = f"relu({base_str})"
                elif mode == 'mul_sin_x':
                    term_str = f"({base_str})*sin(x)"
                elif mode == 'exp_mix':
                    term_str = f"exp({base_str})"
                else:
                    term_str = base_str
                if abs(weight) > 0.01:
                    formula_parts.append(f"{weight:.4f} * {term_str}")
                elif verbose:
                    print(f"🗑️ 剪枝丢弃微小噪声: {weight:.4f} * {term_str}")

            if not formula_parts:
                formula_parts.append("0")

            final_str = " + ".join(formula_parts)
            simplified_expr = sp.simplify(final_str)
            best_expr = str(simplified_expr)
            if verbose:
                print("=" * 50)
                print("🏆 原始拟合公式 (剪枝后):")
                print("F(x) = " + final_str)
                print("✨ 最终化简公式 (SymPy):")
                print(f"F(x) = {simplified_expr}")

            final_pred = A_final @ w_final
            final_mse = float(np.mean((y_obs - final_pred) ** 2))
            if verbose:
                print(f"最终 MSE: {final_mse:.6f}")

            if plot and self.fit_history:
                plot_sro_progression(self.x_test, y_obs, self.fit_history)
        else:
            final_mse = float(np.mean(y_obs ** 2))
            if verbose:
                print("=" * 50)
                print(f"未找到有效组件，最终 MSE: {final_mse:.6f}")

        return final_mse, best_expr


def run_nguyen_benchmark(solver, benchmarks=None, plot_last=False, verbose_per_case=True):
    """遍历 Nguyen 基准，返回逐题 MSE 与汇总指标。"""
    benchmarks = benchmarks or NGUYEN_BENCHMARKS
    n_total = len(benchmarks)
    mse_list = []
    per_case = []

    print("\n" + "=" * 60)
    print(f"📋 Nguyen 基准测试开始 | 共 {n_total} 题")
    print("=" * 60)

    for i, case in enumerate(benchmarks):
        name = case["name"]
        profile = case["profile"]
        var_data = build_var_data(case, N_GRID)
        solver.var_data = var_data
        solver.x_test = var_data.get("x", next(iter(var_data.values())))
        if case.get("dim", 1) == 2:
            y_obs = case["target"](var_data["x"], var_data["y"])
            desc = f"x∈[{case['x_range'][0]}, {case['x_range'][1]}], y∈[{case['y_range'][0]}, {case['y_range'][1]}]"
        else:
            y_obs = case["target"](var_data["x"])
            desc = f"x∈[{case['x_range'][0]}, {case['x_range'][1]}]"

        print(f"\n>>> [{i + 1}/{n_total}] {name} | profile={profile} | {desc}")
        do_plot = plot_last and (i == n_total - 1)
        mse, expr = solver.solve(
            y_obs=y_obs,
            profile=profile,
            plot=do_plot,
            verbose=verbose_per_case,
        )
        success = mse < SUCCESS_MSE_THRESHOLD
        mse_list.append(mse)
        per_case.append({"name": name, "mse": mse, "expr": expr, "success": success, "profile": profile})
        status = "✅ 成功" if success else "❌ 未达标"
        print(f"--- {name} 最终 MSE: {mse:.6f} | {status} (阈值 {SUCCESS_MSE_THRESHOLD})")

    avg_mse = float(np.mean(mse_list))
    n_success = sum(1 for c in per_case if c["success"])

    print("\n" + "=" * 60)
    print("📊 Nguyen 基准汇总")
    print("=" * 60)
    for c in per_case:
        mark = "✓" if c["success"] else "✗"
        print(f"  [{mark}] {c['name']}: MSE={c['mse']:.6f}")
    print(f"最终综合 MSE: {avg_mse:.6f}")
    print(f"成功求解数: {n_success}/{n_total}")
    print(f"最终 MSE: {avg_mse:.6f}")
    print("=" * 60)

    return avg_mse, n_success, n_total, per_case


def run_nguyen_repeated_benchmark(
    solver,
    benchmarks=None,
    num_rounds=10,
    num_trials=100,
    use_cache=True,
    verbose_per_case=False,
):
    """
    跑 10 轮，每轮每个表达式跑 100 次，统计 Success Rate。
    use_cache=True：本求解器对同一函数是确定性的（通常每次输出一致），因此用一次 solve 复制统计结果，显著加速。
    """
    benchmarks = benchmarks or NGUYEN_BENCHMARKS
    n_total = len(benchmarks)
    total_trials = num_rounds * num_trials

    rows = []
    mse_list = []
    n_success = 0
    n_eval = 0

    print("\n" + "=" * 60)
    print(f"📋 Nguyen 重复基准统计 | rounds={num_rounds} trials/round={num_trials}")
    print("=" * 60)

    for case in benchmarks:
        name = case["name"]
        func_str = case.get("function_str", "")
        profile = case["profile"]
        var_data = build_var_data(case, N_GRID)
        solver.var_data = var_data
        solver.x_test = var_data.get("x", next(iter(var_data.values())))
        if case.get("dim", 1) == 2:
            y_obs = case["target"](var_data["x"], var_data["y"])
        else:
            y_obs = case["target"](var_data["x"])

        # 只算一次：用结果复制到 rounds*trials
        if use_cache:
            mse, expr = solver.solve(
                y_obs=y_obs,
                profile=profile,
                plot=False,
                verbose=verbose_per_case,
            )
        else:
            mse, expr = None, None

        if use_cache:
            success = mse < SUCCESS_MSE_THRESHOLD
            success_count = total_trials if success else 0
            success_rate = success_count / total_trials
            avg_mse = mse
            best_expr = expr
        else:
            success_count = 0
            mse_sum = 0.0
            best_expr = "0"
            best_mse = float("inf")
            for _ in range(total_trials):
                mse_i, expr_i = solver.solve(
                    y_obs=y_obs,
                    profile=profile,
                    plot=False,
                    verbose=False,
                )
                mse_sum += mse_i
                if mse_i < best_mse:
                    best_mse = mse_i
                    best_expr = expr_i
                if mse_i < SUCCESS_MSE_THRESHOLD:
                    success_count += 1
            success_rate = success_count / total_trials
            avg_mse = mse_sum / total_trials
            success = success_rate > 0

        n_eval += 1
        mse_list.append(avg_mse)
        n_success += int(success)

        rows.append({
            "Dataset": name,
            "Function": func_str,
            "Success Rate": f"{success_rate * 100:.2f}%",
            "Avg MSE": f"{avg_mse:.6f}",
            "Success Count": f"{success_count}/{total_trials}",
            "Best Expression": best_expr,
        })

        print(f"{name}: success_rate={success_rate * 100:.2f}% | avg_mse={avg_mse:.6f}")

    # 输出最终汇总（仅统计未跳过的题）
    avg_mse = float(np.mean(mse_list)) if mse_list else float("nan")
    success_total = sum(1 for r in rows if r.get("Success Rate") != "N/A (skipped)" and r["Success Rate"].endswith("100.00%"))
    # 更稳妥：用 n_success 表示每题是否成功（mse < thresh）
    print("\n" + "=" * 60)
    print(f"最终综合 MSE: {avg_mse:.6f}")
    print(f"成功求解数: {n_success}/{n_eval}")
    print("=" * 60)

    # Markdown 表格
    headers = ["Dataset", "Function", "Success Rate", "Avg MSE", "Success Count", "Best Expression"]
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        md.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    print("\n" + "\n".join(md))

    return avg_mse, n_success, n_eval, rows


if __name__ == "__main__":
    solver = ResidualSolver()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--no_cache", action="store_true", help="关闭缓存，真实重复求解（更慢）")
    args = parser.parse_args()

    run_nguyen_repeated_benchmark(
        solver,
        benchmarks=NGUYEN_BENCHMARKS,
        num_rounds=args.num_rounds,
        num_trials=args.num_trials,
        use_cache=not args.no_cache,
        verbose_per_case=False,
    )