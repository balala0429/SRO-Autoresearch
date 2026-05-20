import torch
import torch.nn as nn
import torch.nn.functional as F
from tool.BehavioralProjector import BehavioralProjector
from tool.TreeLSTMCell import SymbolicEncoder
from tool.OP_TO_ID import OP_TO_ID
from tool.normalize_y import normalize_y
import random
from tool.Node import Node

class SymbolicAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_points, device):
        super(SymbolicAutoencoder, self).__init__()
        self.vocab_size = vocab_size
        self.device = device

        # 1. Encoder
        self.encoder = SymbolicEncoder(vocab_size, embed_dim, device)
        # 2. Projector
        self.projector = BehavioralProjector(embed_dim, num_points=num_points)
        # 3. Decoder
        self.decoder_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.decoder_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, tree):

        # 直接透传给内部的 encoder (Tree-LSTM)
        return self.encoder(tree)

    def forward_decoder(self, v_h, target_seq, teacher_forcing_ratio=0.5):
        """
        v_h: Encoder 产生的根节点隐藏状态 (Batch, 128)
        target_seq: 目标符号序列 ID (Seq_Len,)
        """
        batch_size = v_h.size(0)
        max_len = target_seq.size(0)

        # 1. 准备 GRU 的初始隐藏状态
        # v_h 是 (Batch, 128)，GRU 期待 (num_layers, Batch, 128)
        hidden = v_h.unsqueeze(0)

        # 2. 准备初始输入 (Start of Sequence)
        # 方案：通常使用 target_seq 的第一个元素作为输入，或者定义一个专门的 SOS 标记
        curr_input_idx = target_seq[0].unsqueeze(0)  # (1,)

        outputs = []

        for t in range(max_len):
            # 将当前索引转为 Embedding: (1, 1, 128)
            # 借用 encoder 的 embedding 层以共享语义
            embedded = self.encoder.embedding(curr_input_idx).unsqueeze(1)

            # GRU 前向计算
            # out: (1, 1, 128), hidden: (1, 1, 128)
            out, hidden = self.decoder_gru(embedded, hidden)

            # 投影到词表空间 (1, vocab_size)
            logits = self.decoder_out(out.squeeze(1))
            outputs.append(logits)

            # 3. Teacher Forcing 逻辑
            is_teacher = torch.rand(1).item() < teacher_forcing_ratio
            if is_teacher and t < max_len - 1:
                # 下一步输入使用真实的标签
                curr_input_idx = target_seq[t + 1].unsqueeze(0)
            else:
                # 下一步输入使用模型刚刚预测出的最高概率分支
                curr_input_idx = logits.argmax(1)

        # 将所有时间步的预测拼接: (1, Seq_Len, Vocab_Size)
        return torch.stack(outputs, dim=1)

    @staticmethod
    def get_tree_seq(node):
        """将树转为 DFS ID 序列"""
        if node is None:
            return [OP_TO_ID['pad']]
        res = [OP_TO_ID[node.op]]
        # 递归处理左右子树
        res += SymbolicAutoencoder.get_tree_seq(node.left)
        res += SymbolicAutoencoder.get_tree_seq(node.right)
        return res



    def get_equivalent_variant(self,node):
        """
        随机返回一个数学等价的变体，用于增强模型的泛化语义。
        """
        if node is None: return None

        # 规则 1: 交换律 (a + b -> b + a, a * b -> b * a)
        if node.op in ['+', '*'] and node.left and node.right:
            if random.random() > 0.5:
                return Node(node.op, left=node.right, right=node.left)

        # 规则 2: 常数合并基础 (x + x -> 2 * x)
        if node.op == '+' and node.left.op == 'x' and node.right.op == 'x':
            return Node('*', left=Node('const', value=2.0), right=Node('x'))

        # 规则 3: 减法的另一种表达 (a - b -> a + (-1 * b))
        # 虽然增加了复杂度，但能让模型理解算子间的联系

        return node  # 如果没有匹配规则，返回原样

    def training_step(self, tree, y_true):
        # 1. 准备原始数据
        y_norm = normalize_y(y_true)
        y_label = torch.tensor(y_norm, dtype=torch.float32).to(self.device)
        seq_list = self.get_tree_seq(tree)
        seq_label = torch.tensor(seq_list, dtype=torch.long).to(self.device)

        # 2. 原始前向传播
        v_h, v_c = self.encoder(tree)
        pred_y = self.projector(v_h)
        pred_seq_logits = self.forward_decoder(v_h, seq_label)

        # 计算基础 Loss
        loss_val = F.mse_loss(pred_y, y_label.view(1, -1))
        loss_seq = F.cross_entropy(
            pred_seq_logits.view(-1, self.vocab_size),
            seq_label.view(-1),
            ignore_index=OP_TO_ID['pad']
        )

        # --- 核心新增：动态语义对齐 (Contrastive Alignment) ---
        loss_sim = torch.tensor(0.0).to(self.device)

        # 随机生成一个等价变体
        variant_tree = self.get_equivalent_variant(tree)

        if variant_tree is not tree:  # 如果成功生成了不同的变体
            # 编码变体树
            v_h_var, _ = self.encoder(variant_tree)
            # 强迫变体向量与原始向量的余弦相似度趋近于 1
            # 使用 (1 - similarity) 作为 Loss
            loss_sim = 1.0 - F.cosine_similarity(v_h, v_h_var).mean()

        # 3. 联合 Loss
        # 给 loss_sim 一个显著的权重（比如 2.0），强行扭转隐空间
        total_loss = loss_val + 0.05 * loss_seq + 2.0 * loss_sim

        return total_loss