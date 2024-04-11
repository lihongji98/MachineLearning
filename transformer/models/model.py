import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_num, trg_vocab_num,
                 embedding_dim,
                 device,
                 max_len=128,
                 stack_num=6, ffn_dim=2048, qkv_dim=64, head_dim=8):
        super().__init__()
        assert embedding_dim == qkv_dim * head_dim, f"{embedding_dim, qkv_dim, head_dim}"

        self.device = device
        self.encoder = Encoder(src_vocab_size=src_vocab_num, x_dim=embedding_dim, stack_num=stack_num, ffn_dim=ffn_dim, qkv_dim=qkv_dim,
                               head_dim=head_dim, max_len=max_len, device=device)
        self.decoder = Decoder(trg_vocab_size=trg_vocab_num, x_dim=embedding_dim, stack_num=stack_num, ffn_dim=ffn_dim, qkv_dim=qkv_dim,
                               head_dim=head_dim, max_len=max_len, device=device)

        self.ffn_nn = nn.Linear(embedding_dim, trg_vocab_num)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, trg):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(N, 1, trg_len, trg_len).to(self.device)

        encoder_x = self.encoder(src, mask=src_mask)
        x = self.decoder(trg, encoder_x, src_mask, trg_mask)

        out = self.ffn_nn(x)

        return out
