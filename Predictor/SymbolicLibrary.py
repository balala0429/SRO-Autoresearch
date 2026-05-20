import faiss
import pickle
import torch
import numpy as np
from tqdm import tqdm


class SymbolicLibrary:
    def __init__(self, encoder, device):
        self.encoder = encoder
        self.device = device
        self.trees = []
        self.vectors = []
        self.index = None

    def build_library(self, generator_func, num_samples=100000):
        print(f"--- 正在生成 {num_samples} 个子树组件 ---")
        self.encoder.eval()

        with torch.no_grad():
            for _ in tqdm(range(num_samples)):
                # 这里的 generator_func 是你之前写的随机树生成逻辑
                tree, _ = generator_func(max_depth=3)
                if tree is None: continue

                # 提取语义向量 v_h
                v_h, _ = self.encoder(tree)

                self.trees.append(tree)
                self.vectors.append(v_h.cpu().numpy().flatten())

        # 转换为 Numpy 矩阵并构建 FAISS 索引
        vector_matrix = np.array(self.vectors).astype('float32')
        # 使用余弦相似度索引 (Inner Product on Normalized Vectors)
        faiss.normalize_L2(vector_matrix)

        self.index = faiss.IndexFlatIP(128)  # 128 是你的隐维度
        self.index.add(vector_matrix)
        print(f"--- 索引构建完成，当前库大小: {self.index.ntotal} ---")

    def search(self, target_vec, k=5):
        """在库中搜索最相似的补丁"""
        target_vec = target_vec.reshape(1, -1).astype('float32')
        faiss.normalize_L2(target_vec)
        distances, indices = self.index.search(target_vec, k)
        return [self.trees[i] for i in indices[0]], distances[0]