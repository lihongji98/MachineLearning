import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, x_dim=512, qkv_dim=64, head_dim=8):
        super().__init__()
        self.input_dim = x_dim
        self.qkv_dim = qkv_dim
        self.head_dim = head_dim
        self.q_nn = nn.Linear(x_dim, qkv_dim * head_dim)
        self.k_nn = nn.Linear(x_dim, qkv_dim * head_dim)
        self.v_nn = nn.Linear(x_dim, qkv_dim * head_dim)

        self.self_attention_nn = nn.Linear(qkv_dim * head_dim, x_dim)

    def forward(self, v, k, q, mask=None):
        batch_size = q.shape[0]
        v_len, k_len, q_len = v.shape[1], k.shape[1], q.shape[1]

        q = self.q_nn(q).reshape(batch_size, q_len, self.head_dim, self.qkv_dim)
        k = self.k_nn(k).reshape(batch_size, k_len, self.head_dim, self.qkv_dim)
        v = self.v_nn(v).reshape(batch_size, v_len, self.head_dim, self.qkv_dim)

        self_attention_score = torch.einsum("bqhd, bkhd->bhqk", [q, k]) / math.sqrt(self.qkv_dim * self.head_dim)

        if mask is not None:
            self_attention_score = self_attention_score.masked_fill(mask == 0, float("-1e9"))

        self_attention_score = torch.softmax(self_attention_score, dim=-1)

        self_attention = torch.einsum("bhql, blhd -> bqhd", [self_attention_score, v]).reshape(
            batch_size, q_len, self.head_dim * self.qkv_dim)

        out = self.self_attention_nn(self_attention)

        return out
