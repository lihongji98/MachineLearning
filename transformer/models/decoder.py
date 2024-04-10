import torch.nn as nn
from .SelfAttention import SelfAttention
from .positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, x_dim=512, stack_num=6, ffn_dim=2048,
                 qkv_dim=64, head_dim=8, dropout=0.1, max_len=128):
        super().__init__()

        self.word_embedding = nn.Embedding(trg_vocab_size, x_dim)
        self.position_embedding = PositionalEncoding(x_dim, max_len)

        self.decoder_stack_layer = nn.ModuleList(
            [DecoderLayer(x_dim=x_dim, ffn_dim=ffn_dim, qkv_dim=qkv_dim, head_dim=head_dim)
             for _ in range(stack_num)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_x, src_mask, trg_mask):
        x = self.dropout((self.word_embedding(x) + self.position_embedding(x)))
        for layer in self.decoder_stack_layer:
            x = layer(x, encoder_x, encoder_x, src_mask, trg_mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, x_dim=512, ffn_dim=2048, qkv_dim=64, head_dim=8, dropout=0.1):
        super().__init__()
        self.masked_self_attention_layer = SelfAttention(x_dim=x_dim, qkv_dim=qkv_dim, head_dim=head_dim)
        self.cross_attention_layer = SelfAttention(x_dim=x_dim, qkv_dim=qkv_dim, head_dim=head_dim)
        self.layerNorm1 = nn.LayerNorm(x_dim)
        self.layerNorm2 = nn.LayerNorm(x_dim)
        self.layerNorm3 = nn.LayerNorm(x_dim)
        self.ffn_nn = nn.ModuleList([nn.Linear(head_dim * qkv_dim, ffn_dim),
                                     nn.ReLU(),
                                     nn.Linear(ffn_dim, head_dim * qkv_dim),
                                     nn.ReLU()])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, v, k, src_mask, trg_mask):
        masked_attention = self.masked_self_attention_layer(x, x, x, trg_mask)
        q = self.dropout(self.layerNorm1(masked_attention + x))
        x = self.layerNorm2(self.cross_attention_layer(v, k, q, src_mask) + q)
        x_res = x
        for layer in self.ffn_nn:
            x = layer(x)
        x = self.dropout(self.layerNorm3(x + x_res))

        return x