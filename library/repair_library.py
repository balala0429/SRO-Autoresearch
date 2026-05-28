"""审计符号库树结构完整性，并重建与有效树对齐的 FAISS 索引。"""
import argparse
import os
import pickle
import sys

import faiss
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool.tree_utils import is_valid_tree, tree_to_str


def repair(index_path, trees_path, out_index=None, out_trees=None):
    index = faiss.read_index(index_path)
    with open(trees_path, "rb") as f:
        trees = pickle.load(f)

    n_total = len(trees)
    valid_trees = []
    vectors = []
    invalid_samples = []

    for i, tree in enumerate(trees):
        if is_valid_tree(tree):
            valid_trees.append(tree)
            vectors.append(index.reconstruct(int(i)))
        else:
            invalid_samples.append((i, tree_to_str(tree) or repr(getattr(tree, "op", "?"))))

    n_bad = n_total - len(valid_trees)
    print(f"总树数: {n_total} | 有效: {len(valid_trees)} | 无效: {n_bad}")
    if invalid_samples[:5]:
        print("无效样例 (最多 5 条):")
        for idx, s in invalid_samples[:5]:
            print(f"  [{idx}] {s}")

    if n_bad == 0:
        print("无需修复。")
        return

    mat = np.array(vectors, dtype="float32")
    faiss.normalize_L2(mat)
    new_index = faiss.IndexFlatIP(mat.shape[1])
    new_index.add(mat)

    out_index = out_index or index_path
    out_trees = out_trees or trees_path
    faiss.write_index(new_index, out_index)
    with open(out_trees, "wb") as f:
        pickle.dump(valid_trees, f)
    print(f"已写入: {out_index} | {out_trees}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--2d", action="store_true", help="修复 2D 库")
    args = parser.parse_args()
    base = "library/library"
    if args.__dict__["2d"]:
        repair(
            f"{base}/symbolic_index_2d.bin",
            f"{base}/symbolic_trees_2d.pkl",
        )
    else:
        repair(
            f"{base}/symbolic_index.bin",
            f"{base}/symbolic_trees.pkl",
        )
