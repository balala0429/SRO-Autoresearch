import torch
import torch.nn.functional as F
import numpy as np

# 导入你的类和工具
from main import NUM_POINTS, get_probing_points
from tool.Node import Node
from tool.OP_TO_ID import OP_TO_ID
from tool.normalize_y import normalize_y
from tool.SymbolicAutoencoder import SymbolicAutoencoder
from Predictor.ResidualPredictor import ResidualPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_predictor():
    print("=" * 50)
    print("      Stage 2: Predictor 雷达探测能力检验      ")
    print("=" * 50)

    # 1. 初始化并加载权重
    x_test = get_probing_points(NUM_POINTS)
    actual_points = len(x_test)

    encoder = SymbolicAutoencoder(
        vocab_size=len(OP_TO_ID), embed_dim=128, num_points=actual_points, device=device
    ).to(device)
    encoder.load_state_dict(torch.load("weight1/encoder_stage1_final.pth"))
    encoder.eval()

    predictor = ResidualPredictor(input_dim=actual_points, output_dim=128).to(device)
    predictor.load_state_dict(torch.load("weight2/predictor_final.pth"))
    predictor.eval()

    # 2. 手动构造几个测试用例 (树结构 + 对应的数值波形)
    test_cases = [
        # 测试 1: 基础周期函数 sin(x)
        ("sin(x)", Node('sin', left=Node('x')), np.sin(x_test)),

        # 测试 2: 指数函数 exp(x) (注意防溢出)
        ("exp(x)", Node('exp', left=Node('x')), np.exp(np.clip(x_test, -10, 10))),

        # 测试 3: 简单的一次函数 2*x
                     # 测试: 最纯粹的孤立节点 x
        ("纯 x", Node('x'), x_test),

        # 测试 4: 稍微复杂的复合 x * sin(x)
        ("x*sin(x)", Node('*', left=Node('x'), right=Node('sin', left=Node('x'))), x_test * np.sin(x_test))
    ]

    # 3. 开始测试
    for name, tree, y_raw in test_cases:
        with torch.no_grad():
            # A. 提取真理坐标 (Label)
            v_target, _ = encoder(tree)

            # B. 提取探测坐标 (Prediction)
            y_norm = normalize_y(y_raw)
            y_input = torch.tensor(y_norm).float().view(1, -1).to(device)
            v_pred = predictor(y_input)

            # C. 计算相似度
            sim = F.cosine_similarity(v_pred, v_target).item()
            distance = F.mse_loss(v_pred, v_target).item()

        print(f"项目: 盲猜补丁 [{name}]")
        print(f" -> 向量方向相似度: {sim:.4f} (大于 0.8 为优秀)")
        print(f" -> 坐标绝对欧氏距离: {distance:.4f}")
        print("-" * 30)


if __name__ == "__main__":
    evaluate_predictor()