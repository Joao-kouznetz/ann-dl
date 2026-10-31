"""Verificações da escala da perda, KL, reparametrização e gradientes."""

import math
import torch
from vae import VAE, loss_terms

torch.manual_seed(42)
x = torch.zeros(2, 784)
prediction = torch.full_like(x, 0.5)
mu = torch.zeros(2, 16)
logvar = torch.zeros_like(mu)
r, k = loss_terms(prediction, x, mu, logvar)
assert abs(r.item() - 784 * math.log(2)) < 0.001 and k.item() == 0
_, k = loss_terms(prediction, x, torch.ones_like(mu), logvar)
assert k.item() == 8
r2, k2 = loss_terms(
    prediction.repeat(2, 1),
    x.repeat(2, 1),
    torch.ones_like(mu).repeat(2, 1),
    logvar.repeat(2, 1),
)
assert abs(r2.item() - r.item()) < 0.001 and k2.item() == 8
model = VAE(16)
inputs = torch.rand(4, 784)
recon, mu, logvar = model(inputs)
a, b = loss_terms(recon, inputs, mu, logvar)
(a + b).backward()
assert recon.shape == (4, 784) and mu.shape == (4, 16)
assert all(
    p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
)
assert model.mu.weight.grad.abs().sum() > 0 and model.logvar.weight.grad.abs().sum() > 0
print("BCE por imagem, KL analítica, médias de batch e gradientes: OK")
