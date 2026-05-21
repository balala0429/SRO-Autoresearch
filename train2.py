import random

import torch
from torch import nn
import numpy as np
import logging
import os
from tqdm import tqdm

# 假设你的模型类和工具函数都在对应的路径
from main import NUM_POINTS, get_probing_points
from tool.Node import generate_safe_tree
from tool.OP_TO_ID import OP_TO_ID
from tool.normalize_y import normalize_y
from tool.SymbolicAutoencoder import SymbolicAutoencoder  # 确保类名一致
from Predictor.ResidualPredictor import ResidualPredictor  # 确保你定义了 Predictor 类

# --- 1. 环境配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# 定义固定的 127 个探测点 (必须与第一阶段完全一致)
x_test = np.linspace(-5, 5, 127)


def train_predictor_stage2(predictor, encoder, num_epochs=100, batch_size=64):
    # --- 2. 准备阶段 ---
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)
    mse_criterion = nn.MSELoss()

    # 锁定 Encoder 参数
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    predictor.to(device)
    encoder.to(device)

    logging.info(f"Start Training Predictor on {device}...")

    for epoch in range(num_epochs):
        predictor.train()
        epoch_loss = 0
        valid_steps = 0

        # 每个 epoch 跑 100 个 iteration
        pbar = tqdm(range(100), desc=f"Epoch {epoch}")
        for _ in pbar:
            batch_v_target = []
            batch_y_input = []

            # 动态构造一个 Batch
            while len(batch_y_input) < batch_size:
                # 修复传参：传入 depth 和固定的 x_test
                current_depth = random.choice([1, 2, 3])
                tree, y_raw = generate_safe_tree(depth=current_depth, x_samples=x_test)

                # 物理安全检查：剔除无效树
                if tree is None or np.any(np.isnan(y_raw)) or np.any(np.isinf(y_raw)):
                    continue
                if np.max(np.abs(y_raw)) > 1e5 or np.var(y_raw) < 1e-6:
                    continue

                # 修改 while 循环内部：
                with torch.no_grad():
                    v_target, _ = encoder(tree)  # 已经在 device 上

                y_norm = normalize_y(y_raw)
                y_input = torch.from_numpy(y_norm).float()  # 先在 CPU

                batch_v_target.append(v_target)  # [ (1, 128), (1, 128), ... ]
                batch_y_input.append(y_input)  # [ (127), (127), ... ]

            # 循环外的合并修改：
            v_target_tensor = torch.cat(batch_v_target, dim=0)  # 已经在 device 上
            y_input_tensor = torch.stack(batch_y_input, dim=0).to(device)  # 一次性上显卡

            # --- 3. 前向传播与优化 ---
            v_pred = predictor(y_input_tensor)

            loss_mse = mse_criterion(v_pred, v_target_tensor)
            # 余弦相似度损失：希望预测向量与目标向量方向一致
            loss_cos = 1.0 - torch.cosine_similarity(v_pred, v_target_tensor).mean()

            # 这里的 2.0 是权重，强化角度对齐（对 FAISS 搜索至关重要）
            loss = loss_mse + 2.0 * loss_cos

            optimizer.zero_grad()
            loss.backward()
            # 加入梯度裁剪保护 Predictor
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            valid_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / valid_steps
        logging.info(f"Epoch {epoch:03d} | Average Loss: {avg_loss:.6f}")

        # 每 10 轮保存一次
        if epoch % 20 == 0:
            torch.save(predictor.state_dict(), f"weight2/predictor_epoch_{epoch}.pth")

    torch.save(predictor.state_dict(), "weight2/predictor_final.pth")
    logging.info("Predictor Training Complete.")


if __name__ == "__main__":

    if not os.path.exists("weight2"):
        os.makedirs("weight2")

    x_test = get_probing_points(NUM_POINTS)
    actual_points = len(x_test)
    encoder = SymbolicAutoencoder(
        vocab_size=len(OP_TO_ID),
        embed_dim=128,
        num_points=actual_points,
        device=device
    )

    encoder.load_state_dict(torch.load("weight1/encoder_stage1_final.pth"))

    predictor = ResidualPredictor(input_dim=actual_points, output_dim=128)

    predictor.load_state_dict(torch.load("weight2/predictor_final.pth"))

    train_predictor_stage2(predictor, encoder, num_epochs=30, batch_size=256)
