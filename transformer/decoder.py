import torch.nn as nn

from SelfAttention import SelfAttention, CrossAttention


class Decoder(nn.Module):
    def __init__(self, x_dim=512, stack_num=6, ffn_dim=2048, qkv_dim=64, head_dim=8):
        super().__init__()
        self.decoder_stack_layer = nn.ModuleList(
            [DecoderLayer(x_dim=x_dim, ffn_dim=ffn_dim, qkv_dim=qkv_dim, head_dim=head_dim)
             for _ in range(stack_num)]
        )

    def forward(self, x, encoder_x, src_mask, trg_mask):
        for layer in self.decoder_stack_layer:
            x = layer(x, encoder_x, src_mask, trg_mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, x_dim=512, ffn_dim=2048, qkv_dim=64, head_dim=8, dropout=0.1):
        super().__init__()
        self.masked_self_attention_layer = SelfAttention(x_dim=x_dim, qkv_dim=qkv_dim, head_dim=head_dim)
        self.cross_attention_layer = CrossAttention(x_dim=x_dim, qkv_dim=qkv_dim, head_dim=head_dim)
        self.layerNorm1 = nn.LayerNorm(x_dim)
        self.layerNorm2 = nn.LayerNorm(x_dim)
        self.layerNorm3 = nn.LayerNorm(x_dim)
        self.ffn_nn = nn.ModuleList([nn.Linear(head_dim * qkv_dim, ffn_dim),
                                     nn.ReLU(),
                                     nn.Linear(ffn_dim, head_dim * qkv_dim),
                                     nn.ReLU()])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_x, src_mask, trg_mask):
        x = self.dropout(self.layerNorm1(self.masked_self_attention_layer(x, trg_mask) + x))
        x = self.layerNorm2(self.cross_attention_layer(x, encoder_x, src_mask) + x)
        x_res = x
        for layer in self.ffn_nn:
            x = layer(x)
        x = self.dropout(self.layerNorm3(x + x_res))

        return x
