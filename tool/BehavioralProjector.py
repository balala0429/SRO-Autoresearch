import torch
import torch.nn as nn
import torch.nn.functional as F

class BehavioralProjector(nn.Module):
    def __init__(self, embed_dim, num_points=32):
        super(BehavioralProjector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_points) # 输出在32个采样点上的预测值
        )

    def forward(self, v_subtree):
        return self.net(v_subtree)