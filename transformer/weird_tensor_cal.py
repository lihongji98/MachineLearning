import torch

q = torch.randn(32, 8, 128, 128)
v = torch.randn(32, 127, 8, 64)

out = torch.einsum("bhqk, bvhd->bvhd", q, v)

print(out.shape)
