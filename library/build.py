import torch
import numpy as np
import faiss
import pickle
import os
from tqdm import tqdm
import random
from main import NUM_POINTS, get_probing_points
from tool.Node import generate_safe_tree
from tool.OP_TO_ID import OP_TO_ID
from tool.SymbolicAutoencoder import SymbolicAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_faiss_library(num_samples=50000, max_depth=3):
    print("=" * 50)
    print(f"   构建 SRO 符号补丁库 (目标数量: {num_samples})   ")
    print("=" * 50)

    # 1. 加载冻结的 Encoder
    x_test = get_probing_points(NUM_POINTS)
    actual_points = len(x_test)

    encoder = SymbolicAutoencoder(
        vocab_size=len(OP_TO_ID), embed_dim=128, num_points=actual_points, device=device
    ).to(device)
    encoder.load_state_dict(torch.load("D:\yjs\SRO_PRO\weight1\encoder_stage1_final.pth"))
    encoder.eval()

    trees = []
    vectors = []

    # 2. 生成子树并提取特征
    print("正在生成树结构并计算语义向量...")
    pbar = tqdm(total=num_samples)

    while len(trees) < num_samples:
        # 核心修改：让弹药库里既有单兵手雷，也有重型导弹
        current_depth = random.choices([1, 2, 3], weights=[0.2, 0.4, 0.4])[0]
        tree, y_raw = generate_safe_tree(depth=current_depth, x_samples=x_test)

        # 严格过滤无效树 (保证库里的弹药都是好用的)
        if tree is None or np.any(np.isnan(y_raw)) or np.any(np.isinf(y_raw)):
            continue
        if np.max(np.abs(y_raw)) > 1e5 or np.var(y_raw) < 1e-6:
            continue

        with torch.no_grad():
            v_target, _ = encoder(tree)
            # 转为 numpy 并压平
            vec_np = v_target.cpu().numpy().flatten()

        trees.append(tree)
        vectors.append(vec_np)
        pbar.update(1)

    pbar.close()

    # 3. 构建 FAISS 索引
    print("正在构建 FAISS 向量索引...")
    vector_matrix = np.array(vectors).astype('float32')

    # 余弦相似度检索的关键：必须对存入的向量进行 L2 归一化
    faiss.normalize_L2(vector_matrix)

    # 使用 Inner Product (内积) 结合归一化，即等价于余弦相似度
    index = faiss.IndexFlatIP(128)
    index.add(vector_matrix)

    # 4. 保存到本地
    if not os.path.exists("library"):
        os.makedirs("library")

    faiss.write_index(index, "library/symbolic_index.bin")
    with open("library/symbolic_trees.pkl", "wb") as f:
        pickle.dump(trees, f)

    print(f"构建完成！成功保存 {index.ntotal} 个符号组件到 library/ 文件夹。")


if __name__ == "__main__":
    # 为了调试速度，你可以先设 num_samples=10000 试试水
    build_faiss_library(num_samples=50000)