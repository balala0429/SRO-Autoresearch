import os
import faulthandler

# 尽量降低 Mac 上 BLAS/OpenMP 与 faiss/torch 的线程冲突概率
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
faulthandler.enable()

import torch
import numpy as np
import faiss
import pickle
import sympy as sp
# 让主程序支持重复评估与表格输出
import argparse
# 导入你项目的模块
from main import NUM_POINTS, get_probing_points
from tool.normalize_y import normalize_y
from tool.Node import Node
from tool.tree_utils import (
    count_active_complexity,
    entry_to_str,
    format_formula,
    get_tree_size,
    is_valid_tree,
    tree_to_str,
)
from Predictor.ResidualPredictor import ResidualPredictor
from tool.visualize.plot_sro_progression import plot_sro_progression
from tool.probing_2d import get_probing_grid_2d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)
if hasattr(faiss, "omp_set_num_threads"):
    faiss.omp_set_num_threads(1)

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
    if node.op == 'pow':
        base = np.abs(eval_tree(node.left, var_data)) + 1e-8
        exp = np.clip(eval_tree(node.right, var_data), -3.0, 3.0)
        return np.power(base, exp)
    return np.zeros_like(base)


def build_var_data(case, n_grid):
    """为 1D/2D 目标函数统一构建变量采样点。"""
    if case.get("dim", 1) == 1:
        x_min, x_max = case["x_range"]
        x = np.linspace(x_min, x_max, n_grid)
        return {'x': x}
    x_min, x_max = case["x_range"]
    y_min, y_max = case["y_range"]
    var_data, _, _ = get_probing_grid_2d(
        nx=NX_2D,
        ny=NY_2D,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
    )
    return var_data


SUCCESS_MSE_THRESHOLD = 1e-4
N_GRID = 127  # 与 predictor 训练及历史实验一致
NX_2D = 32
NY_2D = 32

# 2D 资产路径（重训后会生成）
PREDICTOR_2D_PATH = "weight2_2d/predictor_2d_final.pth"
INDEX_2D_PATH = "library/library/symbolic_index_2d.bin"
TREES_2D_PATH = "library/library/symbolic_trees_2d.pkl"


def simplify_expression_string(expr_str, tiny_coef=1e-8):
    """
    用 SymPy 做后处理化简，并去掉极小系数噪声项，让表达式更简洁。
    """
    x_sym, y_sym = sp.symbols('x y')
    try:
        expr = sp.sympify(expr_str, locals={'x': x_sym, 'y': y_sym})
        expr = sp.expand(expr)
        if isinstance(expr, sp.Add):
            kept_terms = []
            for term in expr.args:
                coeff, _ = term.as_coeff_Mul()
                try:
                    c = float(coeff)
                except Exception:
                    c = None
                if c is None or abs(c) >= tiny_coef:
                    kept_terms.append(term)
            expr = sp.Add(*kept_terms) if kept_terms else sp.Integer(0)
        expr = sp.simplify(expr)
        return str(expr)
    except Exception:
        return expr_str

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
        "beam_limit": 20,
        "penalty_weight": 0.015,
        "mse_tie_ratio": 0.02,
        "max_iters": 14,
        "tol": 1e-4,
        "max_terms": 12,
        "max_tree_nodes": 40,
        "prune_weight": 0.005,
        "prune_mse_slack": 1.05,
        "bootstrap_priors": True,
        "splice_modes": ("raw",),
    },
    "trig": {
        "top_k": 120,
        "beam_limit": 25,
        "penalty_weight": 0.04,
        "mse_tie_ratio": 0.03,
        "max_iters": 25,
        "tol": 1e-4,
        "max_terms": 12,
        "max_tree_nodes": 40,
        "prune_weight": 0.008,
        "prune_mse_slack": 1.08,
        "bootstrap_priors": True,
        "splice_modes": ("raw", "sin", "mul_sin_x", "lin+sin"),
    },
    "log": {
        "top_k": 100,
        "beam_limit": 22,
        "penalty_weight": 0.035,
        "mse_tie_ratio": 0.03,
        "max_iters": 20,
        "tol": 1e-4,
        "max_terms": 10,
        "max_tree_nodes": 36,
        "prune_weight": 0.008,
        "prune_mse_slack": 1.08,
        "bootstrap_priors": False,
        "splice_modes": ("raw", "sin", "exp_mix", "lin+sin"),
    },
    "sqrt": {
        "top_k": 90,
        "beam_limit": 20,
        "penalty_weight": 0.03,
        "mse_tie_ratio": 0.03,
        "max_iters": 18,
        "tol": 1e-4,
        "max_terms": 10,
        "max_tree_nodes": 36,
        "prune_weight": 0.008,
        "prune_mse_slack": 1.08,
        "bootstrap_priors": True,
        "splice_modes": ("raw", "sin", "relu", "mul_sin_x"),
    },
    "multi": {
        "top_k": 140,
        "beam_limit": 30,
        "penalty_weight": 0.04,
        "mse_tie_ratio": 0.03,
        "max_iters": 28,
        "tol": 1e-4,
        "max_terms": 14,
        "max_tree_nodes": 40,
        "prune_weight": 0.008,
        "prune_mse_slack": 1.08,
        "bootstrap_priors": True,
        "splice_modes": ("raw", "sin", "tanh", "mul_sin_x", "mul_sin_y", "lin+sin"),
    },
}

# 按数据套件覆盖 profile 默认（缓解 Extra 上全局 bootstrap 过强、Feynman 上先验不足）
SUITE_PROFILE_OVERRIDES = {
    "nguyen": {
        "bootstrap_mode": "full",
        "prior_queue_cap": None,
    },
    "extra": {
        "bootstrap_mode": "greedy",
        "bootstrap_max_terms": 6,
        "prior_queue_cap": 10,
        "top_k_scale": 1.3,
        "beam_limit_scale": 1.2,
        "max_iters_scale": 1.15,
    },
    "feynman": {
        "bootstrap_mode": "full",
        "prior_queue_cap": None,
        "top_k_scale": 1.15,
        "beam_limit_scale": 1.1,
        "max_iters_scale": 1.2,
    },
    "pmlb": {
        "bootstrap_mode": "greedy",
        "bootstrap_max_terms": 5,
        "prior_queue_cap": 8,
        "top_k_scale": 1.2,
        "beam_limit_scale": 1.1,
    },
    "default": {
        "bootstrap_mode": "full",
        "prior_queue_cap": None,
    },
}


def merge_profile_config(profile: str, suite: str | None = None) -> dict:
    """合并 profile 与 suite 级策略，得到单次 solve 的有效配置。"""
    base = dict(PROFILE_CONFIG.get(profile, PROFILE_CONFIG["trig"]))
    overrides = SUITE_PROFILE_OVERRIDES.get(suite or "default", SUITE_PROFILE_OVERRIDES["default"])
    merged = dict(base)
    for key, val in overrides.items():
        if key.endswith("_scale"):
            continue
        if val is not None:
            merged[key] = val
    for scale_key in ("top_k_scale", "beam_limit_scale", "max_iters_scale"):
        scale = overrides.get(scale_key)
        if not scale:
            continue
        base_key = scale_key.replace("_scale", "")
        if base_key in merged and isinstance(merged[base_key], (int, float)):
            merged[base_key] = int(max(1, round(float(merged[base_key]) * float(scale))))
    if not merged.get("bootstrap_priors", True):
        merged["bootstrap_mode"] = "off"
    return merged


def resolve_suite(case_or_source) -> str:
    """从 benchmark case dict 或 source 字符串解析 suite 名。"""
    if isinstance(case_or_source, dict):
        if case_or_source.get("suite"):
            return str(case_or_source["suite"])
        name = str(case_or_source.get("name", "")).lower()
        src = str(case_or_source.get("source", "")).lower()
        if src in SUITE_PROFILE_OVERRIDES:
            return src
        if "feynman" in name or src == "feynman-sr":
            return "feynman"
        if src == "extra":
            return "extra"
        return "nguyen"
    src = str(case_or_source or "default").lower()
    if src in SUITE_PROFILE_OVERRIDES:
        return src
    if src in ("feynman-sr",):
        return "feynman"
    return "default"


# 流水线消融：五档对比（no prior → full）
PIPELINE_ABLATION_MODES = frozenset(
    {"no_prior", "prior_only", "retrieval_only", "retrieval_plus_splice", "full"}
)


def apply_pipeline_ablation(cfg: dict, mode: str, profile: str) -> dict:
    """
    五档流水线消融（在 merge_profile_config 之后调用）：

    | mode | 结构先验 | Bootstrap | 先验进 Beam | FAISS 检索 | 拼接 |
    |------|---------|-----------|------------|-----------|------|
    | no_prior | ✗ | ✗ | ✗ | ✗ | ✗（常数/零模型基线） |
    | prior_only | ✓ | ✓ | ✗ | ✗ | raw |
    | retrieval_only | ✗ | ✗ | ✗ | ✓ | raw |
    | retrieval_plus_splice | ✗ | ✗ | ✗ | ✓ | profile |
    | full | ✓ | ✓(suite) | ✓ | ✓ | profile |
    """
    if mode not in PIPELINE_ABLATION_MODES:
        raise ValueError(f"unknown pipeline ablation mode: {mode}")
    out = dict(cfg)
    if mode == "no_prior":
        out.update(
            use_profile_priors=False,
            use_bootstrap=False,
            bootstrap_priors=False,
            bootstrap_mode="off",
            use_prior_queue=False,
            use_retrieval=False,
            max_iters=0,
        )
        return out
    if mode == "prior_only":
        out.update(
            use_profile_priors=True,
            use_bootstrap=True,
            bootstrap_priors=True,
            bootstrap_mode=out.get("bootstrap_mode", "full"),
            use_prior_queue=False,
            use_retrieval=False,
            splice_modes=("raw",),
            max_iters=0,
        )
        return out
    if mode == "retrieval_only":
        out.update(
            use_profile_priors=False,
            use_bootstrap=False,
            bootstrap_priors=False,
            bootstrap_mode="off",
            use_prior_queue=False,
            use_retrieval=True,
            splice_modes=("raw",),
        )
        return out
    if mode == "retrieval_plus_splice":
        out.update(
            use_profile_priors=False,
            use_bootstrap=False,
            bootstrap_priors=False,
            bootstrap_mode="off",
            use_prior_queue=False,
            use_retrieval=True,
        )
        return out
    # full
    out.setdefault("use_profile_priors", True)
    out.setdefault("use_bootstrap", True)
    out.setdefault("use_prior_queue", True)
    out.setdefault("use_retrieval", True)
    return out


# --- 主类：残差求解器 ---
class ResidualSolver:
    def __init__(self):
        print("正在加载 SRO 武器系统...")
        self.x_test = get_probing_points(NUM_POINTS)
        self.var_data = {'x': self.x_test}
        actual_points = len(self.x_test)

        # 1) predictor / 2) library：按 profile/输入维度惰性加载
        self._predictor_cache = {}
        self._library_cache = {}
        self.predictor = None
        self.index = None
        self.trees = None
        self.last_diagnostic = {}

    @staticmethod
    def _filter_library(index, trees):
        """丢弃结构不完整的树，并重建与之对齐的 FAISS 索引。"""
        valid_trees = []
        vectors = []
        n_bad = 0
        for i, tree in enumerate(trees):
            if is_valid_tree(tree):
                valid_trees.append(tree)
                vectors.append(index.reconstruct(int(i)))
            else:
                n_bad += 1
        if n_bad == 0:
            return index, trees
        print(f"⚠️ 符号库过滤: 移除 {n_bad}/{len(trees)} 棵结构不完整的树")
        mat = np.array(vectors, dtype="float32")
        faiss.normalize_L2(mat)
        new_index = faiss.IndexFlatIP(mat.shape[1])
        new_index.add(mat)
        return new_index, valid_trees

    def _ensure_assets(self, profile, num_points):
        """根据 profile 与输入维度加载 predictor 与 FAISS 库。"""
        # predictor
        pred_key = (profile, num_points)
        if pred_key not in self._predictor_cache:
            if profile == "multi" and num_points == NX_2D * NY_2D and os.path.exists(PREDICTOR_2D_PATH):
                predictor = ResidualPredictor(input_dim=num_points, output_dim=128).to(device)
                predictor.load_state_dict(torch.load(PREDICTOR_2D_PATH, map_location=torch.device("cpu")))
            else:
                predictor = ResidualPredictor(input_dim=num_points, output_dim=128).to(device)
                predictor.load_state_dict(torch.load("weight2/predictor_final.pth", map_location=torch.device("cpu")))
            predictor.eval()
            self._predictor_cache[pred_key] = predictor

        self.predictor = self._predictor_cache[pred_key]

        # library (FAISS + trees)
        lib_key = "2d" if (profile == "multi" and num_points == NX_2D * NY_2D and os.path.exists(INDEX_2D_PATH)) else "1d"
        if lib_key not in self._library_cache:
            print("正在挂载 FAISS 向量索引与树组件...")
            if lib_key == "2d":
                index_path = INDEX_2D_PATH
                trees_path = TREES_2D_PATH
            else:
                index_path = "library/library/symbolic_index.bin"
                trees_path = "library/library/symbolic_trees.pkl"
            index = faiss.read_index(index_path)
            with open(trees_path, "rb") as f:
                trees = pickle.load(f)
            index, trees = self._filter_library(index, trees)
            self._library_cache[lib_key] = (index, trees)

        self.index, self.trees = self._library_cache[lib_key]

    @staticmethod
    def _build_prior_trees(profile):
        """
        按 profile 注入 **family atoms**（算子族原子基），禁止 target-level / benchmark 级模板。

        原则：只提供单变量/低阶原子（如 sin(x)、x²），不提供已知 benchmark 的完整子结构
        （如 sin(x²)cos(x)、sin(x+x²)、x^y、2sin(x)cos(y) 等）。
        """
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
                x,
                xx,
                Node('sin', x),
                Node('cos', x),
                Node('sin', xx),
            ]
        if profile == "log":
            c1 = Node('const', value=1.0)
            log_x1 = Node('log', Node('+', x, c1))
            log_xx1 = Node('log', Node('+', xx, c1))
            return [
                x,
                xx,
                log_x1,
                log_xx1,
                Node('exp', x),
            ]
        if profile == "sqrt":
            c1 = Node('const', value=1.0)
            return [
                x,
                xx,
                Node('sqrt', x),
                Node('sqrt', Node('+', x, c1)),
            ]
        if profile == "multi":
            y = Node('y')
            yy = Node('*', y, y)
            return [
                x,
                xx,
                y,
                yy,
                Node('sin', x),
                Node('sin', y),
                Node('cos', x),
                Node('cos', y),
                Node('sin', yy),
                Node('*', x, y),
            ]
        return [x, xx]

    def _omp_mse(self, columns, y_obs):
        if not columns:
            return np.mean(y_obs ** 2)
        A = np.column_stack(columns).astype(np.float64, copy=False)
        if not np.all(np.isfinite(A)):
            return float("inf")
        # 限制极端数值，避免 lstsq / matmul 溢出导致不稳定
        A = np.clip(A, -1e6, 1e6)
        y = np.clip(y_obs.astype(np.float64, copy=False), -1e6, 1e6)
        try:
            w, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                pred = A @ w
            if not np.all(np.isfinite(pred)):
                return float("inf")
            return float(np.mean((y - pred) ** 2))
        except np.linalg.LinAlgError:
            return float("inf")

    def _entry_label(self, entry):
        if isinstance(entry, tuple):
            return entry_to_str(entry), entry[1]
        return entry_to_str(entry), 'raw'

    def _build_splice_variants(self, candidate_tree, patch_y, modes):
        # 基础非线性拼接列
        sin_patch = np.sin(patch_y)
        tanh_patch = np.tanh(patch_y)
        relu_patch = np.maximum(patch_y, 0.0)
        mix_sin_x = patch_y * np.sin(self.x_test)
        y_arr = self.var_data.get('y', self.x_test)
        mix_sin_y = patch_y * np.sin(y_arr)
        exp_mix = np.exp(np.clip(patch_y, -8, 8))
        all_variants = {
            "raw": ([patch_y], [(candidate_tree, 'raw')]),
            "sin": ([sin_patch], [(candidate_tree, 'sin')]),
            "tanh": ([tanh_patch], [(candidate_tree, 'tanh')]),
            "relu": ([relu_patch], [(candidate_tree, 'relu')]),
            "mul_sin_x": ([mix_sin_x], [(candidate_tree, 'mul_sin_x')]),
            "mul_sin_y": ([mix_sin_y], [(candidate_tree, 'mul_sin_y')]),
            "exp_mix": ([exp_mix], [(candidate_tree, 'exp_mix')]),
            "lin+sin": (
                [patch_y, sin_patch],
                [(candidate_tree, 'raw'), (candidate_tree, 'sin')],
            ),
        }
        # 简单自适应：根据当前 patch 的方差/振荡水平筛掉明显不合适的拼接
        out = []
        std_val = float(np.std(patch_y))
        # 估计“振荡程度”：零点穿越次数
        sign_changes = int(np.sum(np.sign(patch_y[:-1]) * np.sign(patch_y[1:]) < 0))
        for mode in modes:
            if mode not in all_variants:
                continue
            # 低振荡、近线性残差时，弱化过度非线性拼接
            if mode in ("tanh", "relu", "exp_mix") and sign_changes < 3 and std_val < 0.1:
                continue
            cols, entries = all_variants[mode]
            out.append((cols, entries, mode))
        return out

    def _bootstrap_from_priors(self, prior_trees, y_obs, verbose):
        """多项式/多维题：先把结构先验一次性纳入全局回归，再对残差迭代。"""
        cols, entries = [], []
        for tree in prior_trees:
            if not is_valid_tree(tree):
                continue
            py = eval_tree(tree, self.var_data)
            if np.var(py) < 1e-6 or np.any(np.isnan(py)) or np.any(np.isinf(py)):
                continue
            cols.append(py)
            entries.append((tree, 'raw'))
        if not cols:
            return [], [], float('inf')
        mse = self._omp_mse(cols, y_obs)
        if verbose:
            print(f"📌 先验引导回归: {len(cols)} 项, MSE={mse:.6f}")
        return cols, entries, mse

    def _bootstrap_from_priors_greedy(self, prior_trees, y_obs, verbose, max_terms=6):
        """逐步加入先验列：仅当 MSE 下降才保留（适合 Extra / 表格 OOD）。"""
        cols, entries = [], []
        best_mse = float(np.mean(y_obs ** 2))
        added = 0
        for tree in prior_trees:
            if added >= max_terms:
                break
            if not is_valid_tree(tree):
                continue
            py = eval_tree(tree, self.var_data)
            if np.var(py) < 1e-6 or np.any(np.isnan(py)) or np.any(np.isinf(py)):
                continue
            trial_cols = cols + [py]
            trial_entries = entries + [(tree, "raw")]
            mse = self._omp_mse(trial_cols, y_obs)
            if mse + 1e-15 < best_mse:
                cols, entries = trial_cols, trial_entries
                best_mse = mse
                added += 1
        if verbose and cols:
            print(f"📌 贪心先验引导: {len(cols)} 项, MSE={best_mse:.6f}")
        if not cols:
            return [], [], best_mse
        return cols, entries, best_mse

    @staticmethod
    def prior_only_mse(solver_or_self, prior_trees, y_obs, var_data):
        """仅结构先验 + OMP，不检索库（用于失败诊断）。"""
        solver_or_self.var_data = var_data
        cols = []
        for tree in prior_trees:
            if not is_valid_tree(tree):
                continue
            py = eval_tree(tree, var_data)
            if np.var(py) < 1e-6 or np.any(np.isnan(py)) or np.any(np.isinf(py)):
                continue
            cols.append(py)
        if not cols:
            return float(np.mean(y_obs ** 2))
        return ResidualSolver._omp_mse_static(cols, y_obs)

    @staticmethod
    def _omp_mse_static(columns, y_obs):
        if not columns:
            return float(np.mean(y_obs ** 2))
        A = np.column_stack(columns).astype(np.float64, copy=False)
        A = np.clip(A, -1e6, 1e6)
        y = np.clip(y_obs.astype(np.float64, copy=False), -1e6, 1e6)
        try:
            w, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ w
            if not np.all(np.isfinite(pred)):
                return float("inf")
            return float(np.mean((y - pred) ** 2))
        except np.linalg.LinAlgError:
            return float("inf")

    @staticmethod
    def _finalize_model(active_patches_y, active_trees, y_obs, cfg, verbose):
        """先保留全量拟合精度；仅在 MSE 几乎不变时再剪枝以提升可读性。"""
        if not active_patches_y:
            return [], "0", float(np.mean(y_obs ** 2))

        max_terms = cfg.get("max_terms", 12)
        prune_weight = cfg.get("prune_weight", 0.008)
        prune_slack = cfg.get("prune_mse_slack", 1.08)

        A_full = np.column_stack(active_patches_y).astype(np.float64, copy=False)
        A_full = np.clip(A_full, -1e6, 1e6)
        y_safe = np.clip(y_obs.astype(np.float64, copy=False), -1e6, 1e6)
        w_full, _, _, _ = np.linalg.lstsq(A_full, y_safe, rcond=None)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred_full = A_full @ w_full
        mse_full = float(np.mean((y_safe - pred_full) ** 2)) if np.all(np.isfinite(pred_full)) else float("inf")

        ranked = sorted(
            zip(w_full, active_trees, active_patches_y),
            key=lambda t: abs(t[0]),
            reverse=True,
        )
        kept = [
            (w, e, c)
            for w, e, c in ranked
            if abs(w) >= prune_weight and entry_to_str(e)
        ][:max_terms]
        if not kept:
            kept = list(ranked[: min(3, len(ranked))])

        cols_kept = [t[2] for t in kept]
        entries_kept = [t[1] for t in kept]
        A_kept = np.column_stack(cols_kept).astype(np.float64, copy=False)
        A_kept = np.clip(A_kept, -1e6, 1e6)
        w_kept, _, _, _ = np.linalg.lstsq(A_kept, y_safe, rcond=None)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred_kept = A_kept @ w_kept
        mse_kept = float(np.mean((y_safe - pred_kept) ** 2)) if np.all(np.isfinite(pred_kept)) else float("inf")

        if mse_kept <= mse_full * prune_slack:
            w_out, entries_out, A_out, mse_out = w_kept, entries_kept, A_kept, mse_kept
            if verbose and len(kept) < len(ranked):
                print(f"📐 可读性剪枝: {len(ranked)}→{len(kept)} 项, MSE {mse_full:.6f}→{mse_kept:.6f}")
        else:
            w_out, entries_out, A_out, mse_out = w_full, active_trees, A_full, mse_full
            if verbose:
                print(f"📐 保留全量模型以保证精度 (剪枝后 MSE={mse_kept:.6f} > 阈值)")

        expr = format_formula(w_out, entries_out)
        return entries_out, expr, mse_out

    def solve(
        self,
        y_obs,
        max_iters=None,
        tol=None,
        profile="trig",
        suite=None,
        ablation_mode=None,
        plot=False,
        verbose=True,
        trace=False,
        record_diagnostics=False,
    ):
        cfg = merge_profile_config(profile, suite)
        if ablation_mode and ablation_mode != "full":
            cfg = apply_pipeline_ablation(cfg, ablation_mode, profile)
        if max_iters is None:
            max_iters = cfg["max_iters"]
        if tol is None:
            tol = cfg["tol"]
        use_profile_priors = cfg.get("use_profile_priors", True)
        use_retrieval = cfg.get("use_retrieval", True)

        if ablation_mode == "no_prior":
            baseline_mse = float(np.mean(np.asarray(y_obs, dtype=np.float64) ** 2))
            if verbose:
                print(f"📉 no_prior 基线: MSE={baseline_mse:.6f} (无先验/无检索)")
            if record_diagnostics:
                self.last_diagnostic = {
                    "profile": profile,
                    "suite": suite or "default",
                    "ablation_mode": "no_prior",
                    "final_mse": baseline_mse,
                }
            return baseline_mse, "0"

        # 确保 predictor/索引与当前输入维度一致（1D:127, 2D:32*32）
        if use_retrieval:
            self._ensure_assets(profile=profile, num_points=int(len(y_obs)))
        prior_trees = (
            [t for t in self._build_prior_trees(profile) if is_valid_tree(t)]
            if use_profile_priors
            else []
        )
        diag = {
            "profile": profile,
            "suite": suite or "default",
            "ablation_mode": ablation_mode or "full",
            "bootstrap_mode": cfg.get("bootstrap_mode", "full"),
            "n_prior_trees": len(prior_trees),
            "prior_only_mse": self.prior_only_mse(self, prior_trees, y_obs, self.var_data),
            "bootstrap_mse": None,
            "n_bootstrap_terms": 0,
            "max_faiss_similarity": None,
            "n_active_terms": 0,
            "input_points": int(len(y_obs)),
        }
        max_terms = cfg.get("max_terms", 12)
        max_tree_nodes = cfg.get("max_tree_nodes", 40)
        penalty_weight = cfg.get("penalty_weight", 0.04)
        mse_tie_ratio = cfg.get("mse_tie_ratio", 0.03)

        if verbose:
            print("=" * 50)
            print(
                f"🚀 SRO 推理 | profile={profile} | suite={suite or 'default'} | "
                f"mode={ablation_mode or 'full'} | max_iters={max_iters} tol={tol}"
            )
            print("=" * 50)

        # 记录已被选中的组件（波形和树结构）
        active_patches_y = []
        active_trees = []
        self.fit_history = []
        self.solve_trace = [] if trace else None

        def _trace(event, **kwargs):
            if trace:
                self.solve_trace.append({"event": event, **kwargs})

        _trace(
            "start",
            profile=profile,
            suite=suite or "default",
            ablation_mode=ablation_mode or "full",
            n_prior_atoms=len(prior_trees),
        )

        boot_mode = cfg.get("bootstrap_mode", "full")
        use_bootstrap = (
            bool(cfg.get("use_bootstrap", True))
            and bool(cfg.get("bootstrap_priors", True))
            and boot_mode != "off"
            and use_profile_priors
        )
        if use_bootstrap:
            if boot_mode == "greedy":
                boot_cols, boot_entries, boot_mse = self._bootstrap_from_priors_greedy(
                    prior_trees,
                    y_obs,
                    verbose,
                    max_terms=int(cfg.get("bootstrap_max_terms", 6)),
                )
            else:
                boot_cols, boot_entries, boot_mse = self._bootstrap_from_priors(
                    prior_trees, y_obs, verbose
                )
            diag["bootstrap_mse"] = float(boot_mse)
            diag["n_bootstrap_terms"] = len(boot_cols)
            active_patches_y.extend(boot_cols)
            active_trees.extend(boot_entries)
            if boot_cols:
                _w, _boot_expr, _ = self._finalize_model(
                    active_patches_y, active_trees, y_obs, cfg, verbose=False
                )
                _trace(
                    "bootstrap",
                    n_terms=len(boot_cols),
                    mse=float(boot_mse),
                    terms=[entry_to_str(e) for e in boot_entries],
                    formula=_boot_expr,
                )
            if boot_mse < tol:
                if verbose:
                    print(f"✅ 先验引导已收敛 (MSE: {boot_mse:.6f})")
                _, best_expr, final_mse = self._finalize_model(
                    active_patches_y, active_trees, y_obs, cfg, verbose
                )
                diag["final_mse"] = float(final_mse)
                diag["n_active_terms"] = len(active_trees)
                if record_diagnostics:
                    self.last_diagnostic = diag
                return final_mse, simplify_expression_string(best_expr)

        if not use_retrieval or max_iters <= 0:
            if active_patches_y:
                _, best_expr, final_mse = self._finalize_model(
                    active_patches_y, active_trees, y_obs, cfg, verbose
                )
            else:
                final_mse = float(np.mean(y_obs ** 2))
                best_expr = "0"
            diag["final_mse"] = float(final_mse)
            diag["n_active_terms"] = len(active_trees)
            if record_diagnostics:
                self.last_diagnostic = diag
            if verbose and ablation_mode == "prior_only":
                print(f"✅ prior_only 完成: MSE={final_mse:.6f}")
            return final_mse, simplify_expression_string(best_expr)

        prior_cap = cfg.get("prior_queue_cap")
        use_prior_queue = cfg.get("use_prior_queue", True) and use_profile_priors
        if use_prior_queue:
            prior_for_queue = (
                prior_trees[: int(prior_cap)] if prior_cap is not None else prior_trees
            )
        else:
            prior_for_queue = []

        for step in range(1, max_iters + 1):
            # 1. 计算当前残差
            # 如果有已选组件，用全局系数计算当前最优预测
            if len(active_patches_y) > 0:
                A_current = np.clip(
                    np.column_stack(active_patches_y).astype(np.float64, copy=False),
                    -1e6,
                    1e6,
                )
                y_curr = np.clip(y_obs.astype(np.float64, copy=False), -1e6, 1e6)
                # 全局最小二乘法：自动调整所有已选组件的系数
                w_current, _, _, _ = np.linalg.lstsq(A_current, y_curr, rcond=None)
                y_pred = A_current @ w_current
                if not np.all(np.isfinite(y_pred)):
                    y_pred = np.zeros_like(y_curr)
            else:
                y_pred = np.zeros_like(next(iter(self.var_data.values())))
                w_current = []

            res = y_obs - y_pred
            mse_current = np.mean(res ** 2)

            if mse_current < tol:
                if verbose:
                    print(f"\n✅ 达到收敛精度 (MSE: {mse_current:.6f})，提前结束拟合。")
                _trace("converged", round=step, mse=float(mse_current))
                break

            if verbose:
                print(f"\n--- 第 {step} 轮迭代 | 当前 MSE: {mse_current:.4f} ---")
            _trace("round_start", round=step, residual_mse=float(mse_current), n_active=len(active_trees))

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
            if step == 1 and similarities.size:
                diag["max_faiss_similarity"] = float(np.max(similarities))

            best_patch_cols = None
            best_patch_entries = None
            best_tree = None
            best_mse_pick = float('inf')
            best_complexity_pick = float('inf')
            best_splice_tag = "raw"
            round_candidates = []

            # 4. Beam Search + 多样性去重 + 全局重拟合评估
            if verbose:
                print("雷达锁定候选补丁 (去重后):")
            seen_strs = {self._entry_label(t)[0] for t in active_trees}
            valid_candidates = 0
            candidate_queue = [(self.trees[idx], float(similarities[0][i]))
                               for i, idx in enumerate(indices[0])]
            candidate_queue = [(t, 1.0) for t in prior_for_queue] + candidate_queue
            prior_strs = {tree_to_str(t) for t in prior_for_queue if is_valid_tree(t)}

            for candidate_tree, sim in candidate_queue:
                if not is_valid_tree(candidate_tree):
                    continue
                tree_str = tree_to_str(candidate_tree)
                if not tree_str:
                    continue

                if tree_str in seen_strs:
                    continue
                seen_strs.add(tree_str)
                valid_candidates += 1
                if valid_candidates > beam_limit:
                    break

                patch_y = eval_tree(candidate_tree, self.var_data)
                if np.var(patch_y) < 1e-6 or np.any(np.isnan(patch_y)) or np.any(np.isinf(patch_y)):
                    continue

                # 多样性约束：如果与现有列高度共线，则跳过该候选
                if active_patches_y:
                    denom_new = np.linalg.norm(patch_y)
                    if denom_new < 1e-8:
                        continue
                    max_cos = 0.0
                    for col in active_patches_y:
                        denom_old = np.linalg.norm(col)
                        if denom_old < 1e-8:
                            continue
                        cos_sim = float(np.dot(col, patch_y) / (denom_old * denom_new))
                        max_cos = max(max_cos, abs(cos_sim))
                    # 阈值略小于 1，避免与现有列几乎完全相同
                    if max_cos > 0.995:
                        if verbose:
                            print(f"  跳过候选（与已选列高度相关，cos={max_cos:.4f}）: {tree_str}")
                        continue

                variants = self._build_splice_variants(candidate_tree, patch_y, splice_modes)
                test_mse = float('inf')
                best_variant = None
                best_variant_tag = "raw"
                variant_scores = []
                for cols, entries, vtag in variants:
                    if any(np.any(np.isnan(c)) or np.any(np.isinf(c)) for c in cols):
                        continue
                    mse = self._omp_mse(active_patches_y + cols, y_obs)
                    added = [entry_to_str(e) for e in entries]
                    variant_scores.append({"splice": vtag, "mse": float(mse), "terms": added})
                    if mse < test_mse:
                        test_mse = mse
                        best_variant = (cols, entries)
                        best_variant_tag = vtag

                if best_variant is None:
                    continue

                cand_record = {
                    "candidate": tree_str,
                    "source": "prior" if tree_str in prior_strs else "faiss",
                    "similarity": float(sim),
                    "best_splice": best_variant_tag,
                    "mse": float(test_mse),
                    "variants": variant_scores,
                }
                round_candidates.append(cand_record)

                complexity = get_tree_size(candidate_tree)
                if complexity > max_tree_nodes:
                    continue
                n_new_cols = len(best_variant[0])
                complexity_pick = complexity + 3 * n_new_cols + 0.5 * len(active_patches_y)
                complexity_term = np.log1p(complexity)
                simplicity_term = 1.0 + 0.05 * max(0, n_new_cols - 1)
                adjusted_score = (
                    test_mse
                    * (1.0 + penalty_weight * complexity_term)
                    * simplicity_term
                )
                mse_near_best = test_mse <= best_mse_pick * (1.0 + mse_tie_ratio)

                if verbose:
                    print(
                        f"  [{valid_candidates}] 相似度 {sim:.4f} | 组件: {tree_str} | "
                        f"节点: {complexity} | MSE: {test_mse:.6f} | 调权: {adjusted_score:.6f}")

                pick = False
                if test_mse < best_mse_pick - 1e-15:
                    pick = True
                elif mse_near_best and complexity_pick < best_complexity_pick:
                    pick = True
                if pick:
                    best_mse_pick = test_mse
                    best_complexity_pick = complexity_pick
                    best_patch_cols, best_patch_entries = best_variant
                    best_tree = candidate_tree
                    best_splice_tag = best_variant_tag

            if best_tree is None:
                _trace("round_no_pick", round=step, candidates=round_candidates)
                break
            if len(active_patches_y) + len(best_patch_cols) > max_terms:
                if verbose:
                    print(f"⚠️ 项数已达 {max_terms}，本轮跳过新补丁。")
                continue

            # 5. 正式录用最佳补丁（可含 sin 等非线性列），加入全局列阵
            for col, entry in zip(best_patch_cols, best_patch_entries):
                active_patches_y.append(col)
                active_trees.append(entry)

            A_new = np.clip(
                np.column_stack(active_patches_y).astype(np.float64, copy=False),
                -1e6,
                1e6,
            )
            y_new = np.clip(y_obs.astype(np.float64, copy=False), -1e6, 1e6)
            w_new, _, _, _ = np.linalg.lstsq(A_new, y_new, rcond=None)
            cum_expr = format_formula(w_new, active_trees)
            cum_mse = float(np.mean((y_new - A_new @ w_new) ** 2))
            added_labels = [entry_to_str(e) for e in best_patch_entries]
            _trace(
                "round_pick",
                round=step,
                candidate=tree_to_str(best_tree),
                splice=best_splice_tag,
                added_terms=added_labels,
                pick_mse=float(best_mse_pick),
                cumulative_mse=cum_mse,
                cumulative_formula=cum_expr,
                candidates=round_candidates,
            )
            if verbose:
                print(f"🎯 选定补丁: [{tree_to_str(best_tree)}] splice={best_splice_tag}。全局系数重整中...")
                print(f"   累计 MSE={cum_mse:.6f} | F={cum_expr}")

            # 算一下新阵列的实时系数，仅用于打印展示（w_new 已算）

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
            if not np.all(np.isfinite(current_pred)):
                current_pred = np.zeros_like(y_new)
            current_res = y_new - current_pred
            self.fit_history.append({
                'step': step,
                'patch': tree_to_str(best_tree),
                'y_pred': current_pred,  # OMP 优化后的最新曲线
                'res': current_res,  # 最新的残差
                'mse': np.mean(current_res ** 2)
            })

        # --- 最终公式构建（MSE 用全量/可接受剪枝；SymPy 仅作展示）---
        final_mse = float('inf')
        best_expr = "0"
        if len(active_patches_y) > 0:
            entries_out, final_str, final_mse = self._finalize_model(
                active_patches_y, active_trees, y_obs, cfg, verbose
            )
            n_terms, n_nodes = count_active_complexity(entries_out)
            if verbose:
                print(f"📐 输出公式复杂度: 项数={n_terms}, 基树节点合计={n_nodes}")
            try:
                x_sym, y_sym = sp.symbols('x y')
                simplified_expr = sp.simplify(sp.sympify(final_str, locals={'x': x_sym, 'y': y_sym}))
                best_expr = simplify_expression_string(str(simplified_expr))
            except Exception:
                simplified_expr = final_str
                best_expr = simplify_expression_string(final_str)
            if verbose:
                print("=" * 50)
                print("🏆 拟合公式:")
                print("F = " + final_str)
                print("✨ SymPy 化简 (若可解析):")
                print(f"F = {simplified_expr}")
                print(f"最终 MSE: {final_mse:.6f}")

            if plot and self.fit_history:
                plot_sro_progression(self.x_test, y_obs, self.fit_history)
        else:
            final_mse = float(np.mean(y_obs ** 2))
            if verbose:
                print("=" * 50)
                print(f"未找到有效组件，最终 MSE: {final_mse:.6f}")

        diag["final_mse"] = float(final_mse)
        diag["n_active_terms"] = len(active_trees) if active_patches_y else 0
        _trace("finish", final_mse=float(final_mse), formula=best_expr, n_terms=len(active_trees))
        if record_diagnostics:
            self.last_diagnostic = diag
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
            suite=resolve_suite(case),
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
    ablation_mode=None,
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

        # 只算一次：用结果复制到 rounds*trials（流水线消融请设 use_cache=False）
        if use_cache:
            mse, expr = solver.solve(
                y_obs=y_obs,
                profile=profile,
                suite=resolve_suite(case),
                ablation_mode=ablation_mode,
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
                    suite=resolve_suite(case),
                    ablation_mode=ablation_mode,
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