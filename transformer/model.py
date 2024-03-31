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

        self.encoder = Encoder(x_dim=embedding_dim, stack_num=stack_num, ffn_dim=ffn_dim, qkv_dim=qkv_dim, head_dim=head_dim, x_len=max_len)
        self.mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).to(device)
        self.decoder = Decoder(x_dim=embedding_dim, stack_num=stack_num, ffn_dim=ffn_dim, qkv_dim=qkv_dim, head_dim=head_dim, x_len=max_len, mask=self.mask)

        self.ffn_nn = nn.Linear(embedding_dim, trg_vocab_num)

    def forward(self, src, trg):
        src = self.src_embedding(src)
        trg = self.trg_embedding(trg)

        print(src.shape, trg.shape)
        encoder_x = self.encoder(src)
        x = self.decoder(trg, encoder_x)

        x = self.ffn_nn(x)
        out = torch.softmax(x, dim=-1)

        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tensor1 = torch.tensor([1, 2, 3, 4, 5]).unsqueeze(0).to(device)
    tensor2 = torch.tensor([3, 4, 5, 6, 7]).unsqueeze(0).to(device)
    en = Transformer(src_vocab_num=6, trg_vocab_num=8, embedding_dim=50, stack_num=6, max_len=5, ffn_dim=2048, qkv_dim=10, head_dim=5, device=device).to(device)
    output = en(tensor1, tensor2)
    print(output)
