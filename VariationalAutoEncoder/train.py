import torch
from torch import optim, nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from model import VariationalAutoencoder

epoch = 20
dim_data = 28 * 28
dim_latent = 20
batch_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "")

dataset = MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

model = VariationalAutoencoder(dim_data, dim_latent, conditional=False).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4)


def loss_fn(recon_x, x, _mu, _sigma):
    BCE = nn.BCELoss(reduction='sum')
    BCE_loss = BCE(recon_x, x)
    KL = -torch.sum(1 + torch.log(_sigma.pow(2)) - _mu.pow(2) - _sigma.pow(2))

    return (BCE_loss + KL) / x.size(0)


for epoch in range(epoch):
    loop = tqdm(enumerate(data_loader))
    for i, (data, _) in loop:
        data = data.view(data.shape[0], -1).to(device)
        reconstructed_data, mu, sigma = model(data)

        loss = loss_fn(reconstructed_data, data, mu, sigma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(epoch=epoch+1, loss=loss.item())


model.eval()
for i in range(5):
    epsilon = torch.randn(dim_latent).to(device)
    x = model.decoder(epsilon)
    x = x.view(-1, 1, 28, 28)
    print(x.shape)
    save_image(x, f'res/{i}_generated_digit.png')
