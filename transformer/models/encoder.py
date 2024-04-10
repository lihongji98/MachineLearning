import torch.nn as nn
from .SelfAttention import SelfAttention
from .positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, x_dim=512, stack_num=6, ffn_dim=2048,
                 qkv_dim=64, head_dim=8, x_len=128, dropout=0.1):
        super().__init__()
        self.input_dim = x_dim
        self.x_len = x_len
        self.stack_num = stack_num

        self.word_embedding = nn.Embedding(src_vocab_size, self.input_dim)
        self.position_embedding = PositionalEncoding(self.input_dim, self.x_len)

        self.layer_stack = nn.ModuleList(
            [EncoderLayer(x_dim=x_dim, ffn_dim=ffn_dim, qkv_dim=qkv_dim, head_dim=head_dim)
             for _ in range(stack_num)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(self.word_embedding(x) + self.position_embedding(x))
        for layer in self.layer_stack:
            x = layer(x, x, x, mask=mask)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, x_dim=512, ffn_dim=2048, qkv_dim=64, head_dim=8, dropout=0.1):
        super().__init__()
        self.self_attention_layer = SelfAttention(x_dim=x_dim, qkv_dim=qkv_dim, head_dim=head_dim)
        self.ffn_nn = nn.ModuleList([nn.Linear(x_dim, ffn_dim),
                                     nn.ReLU(),
                                     nn.Linear(ffn_dim, x_dim)
                                     ])
        self.layer_norm1 = nn.LayerNorm(x_dim)
        self.layer_norm2 = nn.LayerNorm(x_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q, mask=None):
        self_attention = self.self_attention_layer(v, k, q, mask=mask)
        x = self.dropout(self.layer_norm1(self_attention) + q)
        x_res = x
        for layer in self.ffn_nn:
            x = layer(x)
        out = self.dropout(self.layer_norm2(x + x_res))

        return out
