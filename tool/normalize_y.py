import numpy as np
import torch


def normalize_y(y):
    # 如果是 Tensor，先转到 CPU 再转成 NumPy
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    # 执行 NumPy 运算
    denom = np.max(np.abs(y)) + 1e-8
    return y / denom