import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim=512, qkv_dim=64, head_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.head_dim = head_dim
        self.q_nn = nn.Linear(input_dim, qkv_dim * head_dim)
        self.k_nn = nn.Linear(input_dim, qkv_dim * head_dim)
        self.v_nn = nn.Linear(input_dim, qkv_dim * head_dim)

        self.self_attention_nn = nn.Linear(qkv_dim * head_dim, input_dim)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.q_nn(x).view(batch_size, -1, self.head_dim, self.qkv_dim)
        k = self.k_nn(x).view(batch_size, -1, self.head_dim, self.qkv_dim)
        v = self.v_nn(x).view(batch_size, -1, self.head_dim, self.qkv_dim)
        self_attention_coefficient = torch.softmax(torch.einsum("blhq, blhk->blh", q, k) / math.sqrt(self.qkv_dim), dim=-1)
        self_attention = torch.einsum("blhv, blh->blhv", v, self_attention_coefficient).view(batch_size, -1, self.head_dim*self.qkv_dim)
        out = self.self_attention_nn(self_attention)

        return out
