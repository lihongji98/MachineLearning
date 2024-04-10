import torch


q = torch.randn(32, 8, 128, 128)
v = torch.randn(32, 127, 8, 64)

out = torch.einsum("bhqk, bvhd->bvhd", q, v)


a = torch.randn(size=(1, 2, 4))
b = torch.softmax(a, dim=-1)
c = torch.softmax(a, dim=-1)[:, -1, :]
d = torch.softmax(a, dim=-1)[:, -1, :].flatten()
top_tokens = torch.argsort(d)[-1:]
print(a)
print(b)
print(c)
print(d)
print(top_tokens)

print("  ====  ")

output = a
print(output)
output = output.permute(1, 0, 2)
print(output)
output = torch.softmax(output, dim=-1)
print(output)

print(output.argmax(2))
print(output.argmax(2)[-1, :])
print(output.argmax(2)[-1, :].item())
