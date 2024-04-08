import math

import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_num, trg_vocab_num,
                 embedding_dim,
                 device,
                 max_len=128,
                 stack_num=6, ffn_dim=2048, qkv_dim=64, head_dim=8):
        super().__init__()
        assert embedding_dim == qkv_dim * head_dim

        self.src_embedding = nn.Embedding(src_vocab_num, embedding_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_num, embedding_dim)
        self.max_len = max_len
        self.device = device

        self.encoder = Encoder(x_dim=embedding_dim, stack_num=stack_num, ffn_dim=ffn_dim, qkv_dim=qkv_dim,
                               head_dim=head_dim)
        self.decoder = Decoder(x_dim=embedding_dim, stack_num=stack_num, ffn_dim=ffn_dim, qkv_dim=qkv_dim,
                               head_dim=head_dim)

        self.ffn_nn = nn.Linear(embedding_dim, trg_vocab_num)

        self.src_positional_encoding = PositionalEncoding(embedding_dim, max_len=max_len)
        self.trg_positional_encoding = PositionalEncoding(embedding_dim, max_len=max_len)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity="relu")

    def forward(self, src, trg):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len), diagonal=0).expand(N, 1, trg_len, trg_len).to(self.device)

        src_word_embedding = self.src_embedding(src)
        src = self.src_positional_encoding(src_word_embedding)

        trg_word_embedding = self.trg_embedding(trg)
        trg = self.trg_positional_encoding(trg_word_embedding)

        encoder_x = self.encoder(src, mask=src_mask)
        x = self.decoder(trg, encoder_x, src_mask, trg_mask)

        out = self.ffn_nn(x)
        # out = torch.softmax(x, dim=-1)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=128):
        super(PositionalEncoding, self).__init__()

        positional_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :]
        return x


if __name__ == "__main__":
    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor1 = torch.randint(0, 100, (5, 128), dtype=torch.int).to(Device)
    tensor2 = torch.randint(0, 100, (5, 128), dtype=torch.int).to(Device)

    en = Transformer(src_vocab_num=100, trg_vocab_num=100,
                     max_len=128,
                     embedding_dim=512, stack_num=6, ffn_dim=2048, qkv_dim=64, head_dim=8, device=Device).to(Device)
    output = en(tensor1, tensor2[:, :-1])
    print(output.shape)
