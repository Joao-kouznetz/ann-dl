"""VAE especificado no enunciado, sem classe pronta de VAE ou KL de biblioteca."""

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )
        self.mu = nn.Linear(256, dimension)
        self.logvar = nn.Linear(256, dimension)
        self.decoder = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)

    def forward(self, x, generator=None):
        mu, logvar = self.encode(x)
        # z = mu + sigma * epsilon: mantém um caminho diferenciável até as heads.
        noise = torch.randn(mu.shape, generator=generator, device=mu.device)
        z = mu + torch.exp(0.5 * logvar) * noise
        return self.decoder(z), mu, logvar


def loss_terms(prediction, x, mu, logvar):
    reconstruction = F.binary_cross_entropy(prediction, x, reduction="sum") / len(x)
    kl = 0.5 * torch.sum(mu.square() + logvar.exp() - 1 - logvar) / len(x)
    return reconstruction, kl
