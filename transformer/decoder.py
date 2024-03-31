import torch.nn as nn

from SelfAttention import SelfAttention, CrossAttention


class Decoder(nn.Module):
    def __init__(self, x_dim=512, stack_num=6, ffn_dim=2048, qkv_dim=64, head_dim=8, x_len=128, mask=None):
        super().__init__()
        self.decoder_stack_layer = nn.ModuleList(
            [DecoderLayer(x_dim=x_dim, ffn_dim=ffn_dim, qkv_dim=qkv_dim, head_dim=head_dim, x_len=x_len, mask=mask)
             for _ in range(stack_num)]
        )

    def forward(self, x, encoder_x):
        for layer in self.decoder_stack_layer:
            x = layer(x, encoder_x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, x_dim=512, ffn_dim=2048, qkv_dim=64, head_dim=8, x_len=128, mask=None):
        super().__init__()
        self.masked_self_attention_layer = SelfAttention(x_dim=x_dim, qkv_dim=qkv_dim, head_dim=head_dim, x_len=x_len)
        self.cross_attention_layer = CrossAttention(x_dim=x_dim, qkv_dim=qkv_dim, head_dim=head_dim, x_len=x_len)
        self.layerNorm1 = nn.LayerNorm(x_dim)
        self.layerNorm2 = nn.LayerNorm(x_dim)
        self.layerNorm3 = nn.LayerNorm(x_dim)
        self.ffn_nn = nn.ModuleList([nn.Linear(head_dim * qkv_dim, ffn_dim),
                                     nn.ReLU(),
                                     nn.Linear(ffn_dim, head_dim * qkv_dim),
                                     nn.ReLU()])
        self.mask = mask

    def forward(self, x, encoder_x):
        x = self.layerNorm1(self.masked_self_attention_layer(x, self.mask) + x)
        x = self.layerNorm2(self.cross_attention_layer(x, encoder_x) + x)
        x_res = x
        for layer in self.ffn_nn:
            x = layer(x)
        x = self.layerNorm3(x + x_res)

        return x
