import argparse
import logging
import os

import numpy as np
import torch

from tool.SymbolicAutoencoder import SymbolicAutoencoder
from tool.Node import generate_safe_tree
from tool.OP_TO_ID import OP_TO_ID
from tool.probing_2d import get_probing_grid_2d


def is_safe_to_train(y):
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        return False
    if np.max(np.abs(y)) > 1e5:
        return False
    if np.max(np.abs(y)) < 1e-7:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--x_min", type=float, default=-1.0)
    parser.add_argument("--x_max", type=float, default=1.0)
    parser.add_argument("--y_min", type=float, default=-1.0)
    parser.add_argument("--y_max", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="weight1_2d")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    var_data, _, _ = get_probing_grid_2d(
        nx=args.nx,
        ny=args.ny,
        x_range=(args.x_min, args.x_max),
        y_range=(args.y_min, args.y_max),
    )
    num_points = len(var_data["x"])

    model = SymbolicAutoencoder(
        vocab_size=len(OP_TO_ID),
        embed_dim=128,
        num_points=num_points,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    logging.info(f"Start 2D stage1 pretrain on {device}. points={num_points} (nx={args.nx}, ny={args.ny})")

    for epoch in range(args.epochs):
        batch = [generate_safe_tree(depth=args.depth, var_data=var_data, allow_y=True) for _ in range(args.batch_size)]
        total_loss = 0.0
        valid = 0

        for tree, y_true in batch:
            if tree is None or not is_safe_to_train(y_true):
                continue

            loss = model.training_step(tree, y_true)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            valid += 1

        if valid:
            logging.info(f"Epoch {epoch:03d} | avg_loss={total_loss/valid:.6f} | valid={valid}/{args.batch_size}")

        if epoch % 10 == 0:
            ckpt = os.path.join(args.out_dir, f"encoder_stage1_2d_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt)

    final_path = os.path.join(args.out_dir, "encoder_stage1_2d_final.pth")
    torch.save(model.state_dict(), final_path)
    logging.info(f"Saved: {final_path}")


if __name__ == "__main__":
    main()

