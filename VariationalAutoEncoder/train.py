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
if_conditional = False

label_num = 10

generated_num = 2

device = torch.device("cuda" if torch.cuda.is_available() else "")

dataset = MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

model = VariationalAutoencoder(dim_data, dim_latent, conditional=if_conditional, label_num=label_num).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4)


def loss_fn(recon_x, _x, _mu, _sigma):
    BCE = nn.BCELoss(reduction='sum')
    BCE_loss = BCE(recon_x, _x)
    KL = -torch.sum(1 + torch.log(_sigma.pow(2)) - _mu.pow(2) - _sigma.pow(2))

    return (BCE_loss + KL) / _x.size(0)


for epoch in range(epoch):
    loop = tqdm(enumerate(data_loader))
    for i, (data, label) in loop:
        # print(data.shape, label.shape)
        #  [128, 1, 28, 28] -> [128, 1, 768],    [128] -> [128, 1, 10]  ===> concat [128, 1, 778]
        data = data.view(data.shape[0], -1).to(device)
        label = label.view(label.shape[0]).to(device)

        reconstructed_data, mu, sigma = model(data, label)

        loss = loss_fn(reconstructed_data, data, mu, sigma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(epoch=epoch+1, loss=loss.item())


model.eval()
for i in range(10):
    epsilon = torch.randn(dim_latent).to(device).unsqueeze(0)
    if if_conditional:
        onehot_label = torch.tensor(generated_num).view(-1).to(device)
        label = nn.functional.one_hot(onehot_label, num_classes=10).view(1, -1)

        latent_variable = torch.cat([epsilon, label], dim=1).view(-1)
    else:
        latent_variable = epsilon.view(-1)
    x = model.decoder(latent_variable)
    x = x.view(-1, 1, 28, 28)
    save_image(x, f'res/VAE/{i}_generated.png')
