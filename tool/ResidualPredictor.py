import torch.nn as nn

class ResidualPredictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 1. 差异提取层 (Cross-Attention)
        # Query 来自目标 V_target，Key/Value 来自当前 V_A
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        # 2. 特征融合层
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)  # 输出预测的补丁向量 V_patch
        )

    def forward(self, v_a, v_target):
        # 增加序列维度以符合 Attention 输入 [Batch, Seq, Dim]
        q = v_target.unsqueeze(1)
        k = v_a.unsqueeze(1)

        # 注意力机制：让目标去“观察”现状哪里不对
        attn_output, _ = self.attention(q, k, k)

        # 通过 MLP 得到补丁特征
        v_patch = self.mlp(attn_output.squeeze(1))
        return v_patch