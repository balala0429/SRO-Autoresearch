import numpy as np
import torch

from tool.evaluate_tree import evaluate_tree
from tool.normalize_y import normalize_y


def solve_residual(y_real, x_points, library, predictor):
    current_f_x = np.zeros_like(y_real)
    max_iters = 5

    for i in range(max_iters):
        # 计算当前残差
        residual = y_real - current_f_x
        if np.linalg.norm(residual) < 1e-3: break

        # 1. 探测：Predictor 给出目标向量
        res_input = torch.tensor(normalize_y(residual)).float()
        v_target = predictor(res_input.view(1, -1))

        # 2. 检索：从 FAISS 库找补丁
        candidate_trees, scores = library.search(v_target.detach().numpy(), k=1)
        patch_tree = candidate_trees[0]

        # 3. 拟合系数：最小二乘法求 alpha (residual = alpha * patch_tree(x))
        y_patch = evaluate_tree(patch_tree, x_points)  # 你需要一个评估函数
        alpha = np.dot(residual, y_patch) / (np.dot(y_patch, y_patch) + 1e-8)

        # 4. 更新公式
        print(f"Step {i}: 发现补丁组件 {alpha:.4f} * {patch_tree}")
        current_f_x += alpha * y_patch