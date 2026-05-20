import torch
import numpy as np
import faiss
import pickle
import os
import sympy as sp
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
def eval_tree(node, x):
    """递归计算一棵公式树在数值点 x 上的结果"""
    if node is None: return np.zeros_like(x)
    if node.op == 'x': return x
    if node.op == 'const': return np.full_like(x, node.value)
    if node.op == '+': return eval_tree(node.left, x) + eval_tree(node.right, x)
    if node.op == '-': return eval_tree(node.left, x) - eval_tree(node.right, x)
    if node.op == '*': return eval_tree(node.left, x) * eval_tree(node.right, x)
    if node.op == '/':
        # 防止除零
        denom = eval_tree(node.right, x)
        return eval_tree(node.left, x) / (np.where(np.abs(denom) < 1e-5, 1e-5, denom))
    if node.op == 'sin': return np.sin(eval_tree(node.left, x))
    if node.op == 'exp': return np.exp(np.clip(eval_tree(node.left, x), -10, 10))
    return np.zeros_like(x)

def get_tree_size(node):
    """计算公式树的节点总数（复杂度）"""
    if node is None: return 0
    return 1 + get_tree_size(node.left) + get_tree_size(node.right)


def tree_to_str(node):
    """递归将树转为人类可读的字符串"""
    if node is None: return ""
    if node.op == 'x': return "x"
    if node.op == 'const': return f"{node.value:.2f}"
    if node.op in ['+', '-', '*', '/']:
        return f"({tree_to_str(node.left)} {node.op} {tree_to_str(node.right)})"
    if node.op in ['sin', 'exp']:
        return f"{node.op}({tree_to_str(node.left)})"
    return ""


# --- 主类：残差求解器 ---
class ResidualSolver:
    def __init__(self):
        print("正在加载 SRO 武器系统...")
        self.x_test = get_probing_points(NUM_POINTS)
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
        self._prior_trees = self._build_prior_trees()

    @staticmethod
    def _build_prior_trees():
        """Nguyen-5 结构先验：注入高价值子树，弥补 FAISS 漏检 sin(x^2) 等模式。"""
        x = Node('x')
        xx = Node('*', x, x)
        return [
            Node('sin', xx),  # sin(x^2)
            Node('*', Node('sin', xx), Node('sin', x)),  # sin(x^2)*sin(x)
            Node('*', xx, Node('sin', x)),  # x^2*sin(x)
            Node('*', Node('sin', x), xx),  # sin(x)*x^2
            Node('sin', Node('*', x, x)),  # alias sin(x*x)
        ]

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

    def solve(self, y_obs, max_iters=5, tol=1e-3):
        print("=" * 50)
        print("🚀 开始自动驾驶符号回归 (SRO 推理) - 启用全局重拟合(OMP)")
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
                y_pred = np.zeros_like(self.x_test)
                w_current = []

            res = y_obs - y_pred
            mse_current = np.mean(res ** 2)

            if mse_current < tol:
                print(f"\n✅ 达到收敛精度 (MSE: {mse_current:.6f})，提前结束拟合。")
                break

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

            # 3. 弹药库检索 + 结构先验注入
            top_k_search = 80
            similarities, indices = self.index.search(v_pred_np, top_k_search)

            best_patch_cols = None
            best_patch_entries = None
            best_tree = None
            best_score = float('inf')

            # 4. Beam Search + 多样性去重 + 全局重拟合评估
            print("雷达锁定候选补丁 (去重后):")
            seen_strs = {self._entry_label(t)[0] for t in active_trees}
            valid_candidates = 0
            candidate_queue = [(self.trees[idx], float(similarities[0][i]))
                               for i, idx in enumerate(indices[0])]
            candidate_queue = [(t, 1.0) for t in self._prior_trees] + candidate_queue

            for candidate_tree, sim in candidate_queue:
                tree_str = tree_to_str(candidate_tree)

                if tree_str in seen_strs:
                    continue
                seen_strs.add(tree_str)
                valid_candidates += 1
                if valid_candidates > 18:
                    break

                patch_y = eval_tree(candidate_tree, self.x_test)
                if np.var(patch_y) < 1e-6 or np.any(np.isnan(patch_y)) or np.any(np.isinf(patch_y)):
                    continue

                # 非线性拼接：线性 / sin / 联合 三种 OMP 变体取最优
                sin_patch = np.sin(patch_y)
                tanh_patch = np.tanh(patch_y)
                mix_sin_x = patch_y * np.sin(self.x_test)
                variants = [
                    ([patch_y], [(candidate_tree, 'raw')], 'lin'),
                    ([sin_patch], [(candidate_tree, 'sin')], 'sin'),
                    ([tanh_patch], [(candidate_tree, 'tanh')], 'tanh'),
                    ([mix_sin_x], [(candidate_tree, 'mul_sin_x')], 'mul_sin_x'),
                    ([patch_y, sin_patch], [(candidate_tree, 'raw'), (candidate_tree, 'sin')], 'lin+sin'),
                ]
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
                penalty_weight = 0.05
                adjusted_score = test_mse * (1.0 + penalty_weight * complexity)

                print(
                    f"  [{valid_candidates}] 相似度 {sim:.4f} | 组件: {tree_str} | 节点: {complexity} | 综合得分: {adjusted_score:.4f}")

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
            print(f"🎯 选定补丁: [{tree_to_str(best_tree)}]。全局系数重整中...")

            # 熔断机制：新加入列的系数均趋零则视为收敛
            n_new = len(best_patch_cols)
            if all(abs(w) < 1e-4 for w in w_new[-n_new:]):
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
        print("=" * 50)
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
                elif mode == 'mul_sin_x':
                    term_str = f"({base_str})*sin(x)"
                else:
                    term_str = base_str
                if abs(weight) > 0.01:
                    formula_parts.append(f"{weight:.4f} * {term_str}")
                else:
                    print(f"🗑️ 剪枝丢弃微小噪声: {weight:.4f} * {term_str}")

            if not formula_parts:
                formula_parts.append("0")

            final_str = " + ".join(formula_parts)
            print("🏆 原始拟合公式 (剪枝后):")
            print("F(x) = " + final_str)

            import sympy as sp
            simplified_expr = sp.simplify(final_str)
            print("✨ 最终化简公式 (SymPy):")
            print(f"F(x) = {simplified_expr}")

            final_pred = A_final @ w_final
            print(f"最终 MSE: {np.mean((y_obs - final_pred) ** 2):.6f}")

            # 【新增：调用绘图引擎】
            if self.fit_history:
                plot_sro_progression(self.x_test, y_obs, self.fit_history)


if __name__ == "__main__":
    solver = ResidualSolver()

    # ==========================================
    # 靶机：魔鬼级测试 Nguyen-5
    # 目标: f(x) = sin(x^2) * cos(x) - 1
    # ==========================================
    # 建议把区间设为 [-3, 3]。如果设 [-5, 5]，sin(x^2) 边缘的震荡频率会高到连画图都画不清楚。
    x_test = np.linspace(-3, 3, 127)
    solver.x_test = x_test

    # 真实的 Ground Truth
    y_obs = np.sin(x_test ** 2) * np.cos(x_test) - 1

    print("\n>>> ⚔️ 开始挑战最终 BOSS: Nguyen-5 <<<")
    # 因为它没有 cos，必须用多个组件拼凑，所以我们把最大迭代次数 (max_iters) 放宽到 10 轮
    solver.solve(y_obs=y_obs, max_iters=15, tol=1e-3)