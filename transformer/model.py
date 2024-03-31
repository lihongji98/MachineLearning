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

        self.encoder = Encoder(x_dim=embedding_dim, stack_num=stack_num, ffn_dim=ffn_dim, qkv_dim=qkv_dim,
                               head_dim=head_dim, x_len=max_len)
        self.mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).to(device)
        self.decoder = Decoder(x_dim=embedding_dim, stack_num=stack_num, ffn_dim=ffn_dim, qkv_dim=qkv_dim,
                               head_dim=head_dim, x_len=max_len, mask=self.mask)

        self.ffn_nn = nn.Linear(embedding_dim, trg_vocab_num)

        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=max_len, dropout=0.1)

    def forward(self, src, trg):
        src = self.positional_encoding(self.src_embedding(src))
        trg = self.positional_encoding(self.trg_embedding(trg))

        encoder_x = self.encoder(src)
        x = self.decoder(trg, encoder_x)

        x = self.ffn_nn(x)
        out = torch.softmax(x, dim=-1)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":
    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(Device)
    tensor1 = torch.tensor([1, 2, 3, 4, 5]).unsqueeze(0).to(Device)
    tensor2 = torch.tensor([3, 4, 5, 6, 7]).unsqueeze(0).to(Device)
    en = Transformer(src_vocab_num=6, trg_vocab_num=8,
                     max_len=5,
                     embedding_dim=80, stack_num=6, ffn_dim=2048, qkv_dim=10, head_dim=8, device=Device).to(Device)
    output = en(tensor1, tensor2)
    print(output)
