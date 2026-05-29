import argparse
import logging
import os
import random

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from tool.Node import generate_safe_tree
from tool.OP_TO_ID import OP_TO_ID
from tool.SymbolicAutoencoder import SymbolicAutoencoder
from tool.normalize_y import normalize_y
from tool.probing_2d import get_probing_grid_2d
from Predictor.ResidualPredictor import ResidualPredictor


def train_predictor_2d(
    predictor,
    encoder,
    var_data,
    num_epochs=30,
    batch_size=256,
    iters_per_epoch=100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.to(device)
    encoder.to(device)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)
    mse_criterion = nn.MSELoss()

    logging.info(f"Start 2D predictor training on {device}...")

    for epoch in range(num_epochs):
        predictor.train()
        epoch_loss = 0.0
        valid_steps = 0
        pbar = tqdm(range(iters_per_epoch), desc=f"Epoch {epoch}")
        for _ in pbar:
            batch_v_target = []
            batch_y_input = []

            while len(batch_y_input) < batch_size:
                depth = random.choice([1, 2, 3])
                tree, y_raw = generate_safe_tree(depth=depth, var_data=var_data, allow_y=True)
                if tree is None:
                    continue
                if np.any(np.isnan(y_raw)) or np.any(np.isinf(y_raw)):
                    continue
                if np.max(np.abs(y_raw)) > 1e5 or np.var(y_raw) < 1e-6:
                    continue

                with torch.no_grad():
                    v_target, _ = encoder(tree)  # (1, 128) on device

                y_norm = normalize_y(y_raw)
                y_input = torch.from_numpy(y_norm).float()  # CPU

                batch_v_target.append(v_target)
                batch_y_input.append(y_input)

            v_target_tensor = torch.cat(batch_v_target, dim=0)  # (B,128) on device
            y_input_tensor = torch.stack(batch_y_input, dim=0).to(device)  # (B,N)

            v_pred = predictor(y_input_tensor)
            loss_mse = mse_criterion(v_pred, v_target_tensor)
            loss_cos = 1.0 - torch.cosine_similarity(v_pred, v_target_tensor).mean()
            loss = loss_mse + 2.0 * loss_cos

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            valid_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        logging.info(f"Epoch {epoch:03d} | avg_loss={epoch_loss/max(valid_steps,1):.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--x_min", type=float, default=-1.0)
    ap.add_argument("--x_max", type=float, default=1.0)
    ap.add_argument("--y_min", type=float, default=-1.0)
    ap.add_argument("--y_max", type=float, default=1.0)
    ap.add_argument("--encoder_ckpt", type=str, default="weight1_2d/encoder_stage1_2d_final.pth")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--iters_per_epoch", type=int, default=100)
    ap.add_argument("--out_dir", type=str, default="weight2_2d")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    var_data, _, _ = get_probing_grid_2d(nx=args.nx, ny=args.ny, x_range=(args.x_min, args.x_max), y_range=(args.y_min, args.y_max))
    num_points = len(var_data["x"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SymbolicAutoencoder(vocab_size=len(OP_TO_ID), embed_dim=128, num_points=num_points, device=device)
    encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device))

    predictor = ResidualPredictor(input_dim=num_points, output_dim=128)

    train_predictor_2d(
        predictor=predictor,
        encoder=encoder,
        var_data=var_data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        iters_per_epoch=args.iters_per_epoch,
    )

    final_path = os.path.join(args.out_dir, "predictor_2d_final.pth")
    torch.save(predictor.state_dict(), final_path)
    logging.info(f"Saved: {final_path}")


if __name__ == "__main__":
    main()

