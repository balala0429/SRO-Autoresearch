import torch
import torch.nn as nn

from tool.normalize_y import normalize_y


class ResidualPredictor(nn.Module):
    def __init__(self, input_dim=127, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # 输入是 127 个采样点的残差值
        return self.net(x)

