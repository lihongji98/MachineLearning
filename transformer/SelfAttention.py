import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, x_dim=512, qkv_dim=64, head_dim=8, x_len=128):
        super().__init__()
        self.input_dim = x_dim
        self.x_len = x_len
        self.qkv_dim = qkv_dim
        self.head_dim = head_dim
        self.q_nn = nn.Linear(x_dim, qkv_dim * head_dim)
        self.k_nn = nn.Linear(x_dim, qkv_dim * head_dim)
        self.v_nn = nn.Linear(x_dim, qkv_dim * head_dim)

        self.self_attention_nn = nn.Linear(qkv_dim * head_dim, x_dim)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        q = self.q_nn(x).view(batch_size, self.x_len, self.head_dim, self.qkv_dim)
        k = self.k_nn(x).view(batch_size, self.x_len, self.head_dim, self.qkv_dim)
        v = self.v_nn(x).view(batch_size, self.x_len, self.head_dim, self.qkv_dim)

        self_attention_score = torch.softmax(torch.einsum("bqhd, bkhd->bhqk", q, k) / math.sqrt(self.qkv_dim), dim=-1)

        if mask is not None:
            self_attention_score = torch.where(mask == 1, float(1e-20), self_attention_score)

        self_attention = torch.einsum("bvhd, bhqk->bvhd", v, self_attention_score).view(batch_size, -1, self.head_dim * self.qkv_dim)

        out = self.self_attention_nn(self_attention)

        return out


class CrossAttention(nn.Module):
    def __init__(self, x_dim=512, qkv_dim=64, head_dim=8, x_len=128):
        super().__init__()
        self.input_dim = x_dim
        self.x_len = x_len
        self.qkv_dim = qkv_dim
        self.head_dim = head_dim
        self.q_nn = nn.Linear(x_dim, qkv_dim * head_dim)
        self.k_nn = nn.Linear(x_dim, qkv_dim * head_dim)
        self.v_nn = nn.Linear(x_dim, qkv_dim * head_dim)

        self.self_attention_nn = nn.Linear(qkv_dim * head_dim, x_dim)

    def forward(self, x, encoder_x):
        batch_size = x.size(0)
        q = self.q_nn(encoder_x).view(batch_size, self.x_len, self.head_dim, self.qkv_dim)
        k = self.k_nn(encoder_x).view(batch_size, self.x_len, self.head_dim, self.qkv_dim)
        v = self.v_nn(x).view(batch_size, self.x_len, self.head_dim, self.qkv_dim)

        cross_attention_score = torch.softmax(torch.einsum("bqhd, bkhd->bhqk", q, k) / math.sqrt(self.qkv_dim), dim=-1)

        cross_attention = torch.einsum("bvhd, bhqk->bvhd", v, cross_attention_score).view(batch_size, -1, self.head_dim * self.qkv_dim)

        out = self.self_attention_nn(cross_attention)

        return out
