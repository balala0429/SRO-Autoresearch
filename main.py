import torch
from tool.SymbolicAutoencoder import SymbolicAutoencoder
import numpy as np
from tool.Node import generate_random_tree,generate_safe_tree,Node
import logging
import torch.nn.functional as F
import os
from tool.OP_TO_ID import OP_TO_ID
from tool.normalize_y import normalize_y

# --- 1. 配置日志系统 ---
log_file = "pretrain_stage1.log"

# 创建一个处理器用于输出到控制台
console_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO, # 这里才是放 logging.INFO 的地方
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file), # 保存到文件
        console_handler                # 输出到控制台
    ]
)


def tree_to_data(node):
    """将树转化为递归训练所需的字典格式"""
    if node is None:
        return {'op': OP_TO_ID['pad'], 'left': None, 'right': None}
    return {
        'op': OP_TO_ID[node.op],
        'left': tree_to_data(node.left),
        'right': tree_to_data(node.right)
    }


# --- 核心配置 ---
NUM_POINTS = 128  # 采样点数量，必须与 Projector 的输出维度一致
X_RANGE = (-5, 5)

def get_probing_points(num_points=128):
    """
    生成非均匀分布的采样点：中间密，两边疏
    """
    # 在核心区 [-1, 1] 密集采样 (40%)
    center = np.linspace(-1, 1, int(num_points * 0.4))
    # 在边缘区 [-5, -1] 和 [1, 5] 稀疏采样 (60%)
    left = np.linspace(-5, -1.05, int(num_points * 0.3))
    right = np.linspace(1.05, 5, int(num_points * 0.3))

    x = np.concatenate([left, center, right])
    return np.sort(x).astype(np.float32)

def is_safe_to_train(y):
    # 1. 检查是否有无效值
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        return False
    # 2. 检查数值范围：超过 1e5 的通常是爆炸函数，直接弃用
    if np.max(np.abs(y)) > 1e5:
        return False
    # 3. 检查是否有意义：全 0 的函数没法提供梯度
    if np.max(np.abs(y)) < 1e-7:
        return False
    return True



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test = get_probing_points(NUM_POINTS)
    actual_points = len(x_test)

    # 2. 实例化模型并加载权重
    model = SymbolicAutoencoder(
        vocab_size=len(OP_TO_ID),
        embed_dim=128,
        num_points=actual_points,
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logging.info(f"Start pre-training on {device}. Probing points: {actual_points}")

    for epoch in range(500):
        # 记录每个 epoch 的平均 loss
        batch_data = [generate_safe_tree(depth=2, x_samples=x_test) for _ in range(256)]
        total_loss = 0
        valid_count = 0

        # --- 循环中 ---
        for tree, y_true in batch_data:
            if tree is None or not is_safe_to_train(y_true):
                continue  # 遇到“毒树”，直接跳过

            loss = model.training_step(tree, y_true)

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("Catch a NaN/Inf loss! Skipping this step to protect weights.")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            valid_count += 1

        # --- 2. 打印并保存 Loss ---
        if valid_count > 0:
            avg_loss = total_loss / valid_count
            logging.info(f"Epoch {epoch:03d} | Average Loss: {avg_loss:.8f} | Valid Trees: {valid_count}/256")

        # --- 3. 定期保存模型权重 (防止程序中断) ---
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"encoder_stage1_epoch_{epoch}.pth")
            logging.info(f"Model checkpoint saved at epoch {epoch}")

    # 最终保存
    torch.save(model.state_dict(), "weight1/encoder_stage1_final.pth")
    logging.info("Training complete. Final model saved.")






