import argparse
import os
import pickle
import random
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import torch
from tqdm import tqdm

# allow running as script from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool.Node import generate_safe_tree
from tool.OP_TO_ID import OP_TO_ID
from tool.tree_utils import is_valid_tree
from tool.SymbolicAutoencoder import SymbolicAutoencoder
from tool.probing_2d import get_probing_grid_2d


def build_faiss_library_2d(
    num_samples=50000,
    max_depth=3,
    nx=32,
    ny=32,
    x_min=-1.0,
    x_max=1.0,
    y_min=-1.0,
    y_max=1.0,
    encoder_ckpt="weight1_2d/encoder_stage1_2d_final.pth",
    out_dir="library/library",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    var_data, _, _ = get_probing_grid_2d(nx=nx, ny=ny, x_range=(x_min, x_max), y_range=(y_min, y_max))
    num_points = len(var_data["x"])

    encoder = SymbolicAutoencoder(
        vocab_size=len(OP_TO_ID),
        embed_dim=128,
        num_points=num_points,
        device=device,
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    encoder.eval()

    trees = []
    vectors = []

    print("=" * 50)
    print(f"构建 2D 符号补丁库 | samples={num_samples} | depth<= {max_depth} | grid={nx}x{ny} (points={num_points})")
    print("=" * 50)

    pbar = tqdm(total=num_samples)
    while len(trees) < num_samples:
        current_depth = random.choices(list(range(1, max_depth + 1)), weights=[0.2, 0.4, 0.4][:max_depth])[0]
        tree, y_raw = generate_safe_tree(depth=current_depth, var_data=var_data, allow_y=True)

        if tree is None or not is_valid_tree(tree):
            continue
        if np.any(np.isnan(y_raw)) or np.any(np.isinf(y_raw)):
            continue
        if np.max(np.abs(y_raw)) > 1e5 or np.var(y_raw) < 1e-6:
            continue

        with torch.no_grad():
            v_target, _ = encoder(tree)
            vec_np = v_target.detach().cpu().numpy().flatten().astype("float32")

        trees.append(tree)
        vectors.append(vec_np)
        pbar.update(1)

    pbar.close()

    vector_matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vector_matrix)
    index = faiss.IndexFlatIP(128)
    index.add(vector_matrix)

    index_path = os.path.join(out_dir, "symbolic_index_2d.bin")
    trees_path = os.path.join(out_dir, "symbolic_trees_2d.pkl")
    faiss.write_index(index, index_path)
    with open(trees_path, "wb") as f:
        pickle.dump(trees, f)

    print(f"Saved index: {index_path}")
    print(f"Saved trees: {trees_path}")
    print(f"Total: {index.ntotal}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=50000)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--x_min", type=float, default=-1.0)
    ap.add_argument("--x_max", type=float, default=1.0)
    ap.add_argument("--y_min", type=float, default=-1.0)
    ap.add_argument("--y_max", type=float, default=1.0)
    ap.add_argument("--encoder_ckpt", type=str, default="weight1_2d/encoder_stage1_2d_final.pth")
    ap.add_argument("--out_dir", type=str, default="library/library")
    args = ap.parse_args()

    build_faiss_library_2d(
        num_samples=args.num_samples,
        max_depth=args.max_depth,
        nx=args.nx,
        ny=args.ny,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        encoder_ckpt=args.encoder_ckpt,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

