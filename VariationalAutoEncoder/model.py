import torch
import torch.nn as nn


class VariationalAutoencoder(nn.Module):
    def __init__(self, dim_data, dim_latent, conditional=False, label_num=None):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(dim_data, dim_latent, conditional, label_num)
        self.decoder = Decoder(dim_data, dim_latent, conditional, label_num)
        self.conditional = conditional

    @staticmethod
    def reparameterize(mu, sigma):
        epsilon = torch.randn_like(mu)
        z_reparameterized = mu + epsilon * sigma

        return z_reparameterized

    def forward(self, data, label=None):
        if self.conditional:
            onehot_label = nn.functional.one_hot(label, num_classes=10)
            mu, sigma = self.encoder(data, onehot_label)
            z = self.reparameterize(mu, sigma)
            z = torch.cat([z, onehot_label], dim=1)
        else:
            mu, sigma = self.encoder(data)
            z = self.reparameterize(mu, sigma)

        reconstructed_data = self.decoder(z)

        return reconstructed_data, mu, sigma


class Encoder(nn.Module):
    def __init__(self, dim_data, dim_latent, conditional, label_num=None):
        super(Encoder, self).__init__()
        self.dim_data = dim_data + label_num if conditional else dim_data
        self.dim_latent = dim_latent
        self.dim_hid1 = 256
        self.dim_hid2 = 64
        self.conditional = conditional

        self.MLP = nn.ModuleList([
            nn.Linear(self.dim_data, self.dim_hid1),
            nn.ReLU(),
            nn.Linear(self.dim_hid1, self.dim_hid2),
            nn.ReLU()
        ])
        self.nn_mu = nn.Linear(self.dim_hid2, self.dim_latent)
        self.nn_sigma = nn.Linear(self.dim_hid2, self.dim_latent)

    def forward(self, data, onehot_label=None):
        if self.conditional:
            data = torch.cat([data, onehot_label], dim=-1)

        for layer in self.MLP:
            data = layer(data)
        mu = self.nn_mu(data)
        sigma = self.nn_sigma(data)

        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, dim_data, dim_latent, conditional, label_num=None):
        super(Decoder, self).__init__()
        self.dim_data = dim_data
        self.dim_latent = dim_latent + label_num if conditional else dim_latent
        self.dim_hid1 = 64
        self.dim_hid2 = 256

        self.MLP = nn.ModuleList([
            nn.Linear(self.dim_latent, self.dim_hid1),
            nn.ReLU(),
            nn.Linear(self.dim_hid1, self.dim_hid2),
            nn.ReLU(),
            nn.Linear(self.dim_hid2, self.dim_data),
            nn.Sigmoid()
        ])

    def forward(self, latent_var):
        for layer in self.MLP:
            latent_var = layer(latent_var)
        reconstructed_data = latent_var

        return reconstructed_data

