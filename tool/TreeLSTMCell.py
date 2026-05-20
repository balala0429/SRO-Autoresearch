import torch
import torch.nn as nn
import torch.nn.functional as F

class TreeLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TreeLSTMCell, self).__init__()
        # 针对二叉树设计：左孩子、右孩子、当前节点Embedding
        self.W_i = nn.Linear(input_dim + 2 * hidden_dim, hidden_dim)
        self.W_f_left = nn.Linear(hidden_dim, hidden_dim)
        self.W_f_right = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim + 2 * hidden_dim, hidden_dim)
        self.W_u = nn.Linear(input_dim + 2 * hidden_dim, hidden_dim)

    def forward(self, x, h_left, c_left, h_right, c_right):
        # 拼接当前节点信息与子节点隐藏状态
        combined = torch.cat([x, h_left, h_right], dim=-1)

        i = torch.sigmoid(self.W_i(combined))
        o = torch.sigmoid(self.W_o(combined))
        u = torch.tanh(self.W_u(combined))

        # 遗忘门：分别针对左右孩子
        f_left = torch.sigmoid(self.W_f_left(h_left))
        f_right = torch.sigmoid(self.W_f_right(h_right))

        c = i * u + f_left * c_left + f_right * c_right
        h = o * torch.tanh(c)
        return h, c



OP_TO_ID = {'+': 0, '-': 1, '*': 2, '/': 3, 'sin': 4, 'exp': 5, 'x': 6, 'const': 7, 'pad': 8}


class SymbolicEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, device):
        super(SymbolicEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 新增：专门处理常数值的线性层，将 (128+1) 压回 128
        self.const_fc = nn.Linear(embed_dim + 1, embed_dim)

        self.cell = TreeLSTMCell(embed_dim, embed_dim)
        self.null_h = nn.Parameter(torch.zeros(1, embed_dim))
        self.null_c = nn.Parameter(torch.zeros(1, embed_dim))
        self.device = device

    def forward(self, node):
        op_idx = OP_TO_ID[node.op]
        # 1. 处理输入向量 x
        if node.op == 'const':
            # 基础 embedding (128维)
            op_emb = self.embedding(torch.tensor([op_idx], device=self.device))
            # 具体数值 (1维)
            val_feat = torch.tensor([[node.value]], device=self.device).float()
            # 拼接并压缩回 128维
            x = self.const_fc(torch.cat([op_emb, val_feat], dim=-1))
        else:
            # 普通算子直接 lookup (128维)
            x = self.embedding(torch.tensor([op_idx], device=self.device))

        # 2. 递归获取子节点状态
        if node.left is None and node.right is None:
            # 叶子节点直接计算 (x, pad, pad)
            return self.cell(x, self.null_h, self.null_c, self.null_h, self.null_c)

        h_l, c_l = self.forward(node.left) if node.left else (self.null_h, self.null_c)
        h_r, c_r = self.forward(node.right) if node.right else (self.null_h, self.null_c)

        # 3. 融合
        return self.cell(x, h_l, c_l, h_r, c_r)


