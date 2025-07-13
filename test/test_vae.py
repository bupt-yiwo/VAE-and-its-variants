import torch
import unittest
from models.vanilla_vae import VanillaVAE


model = VanillaVAE(3,10)
out = model(torch.randn(16,3,64,64))
print(out[0].size())
loss = model.loss_function(*out, M_N=0.005)
print(loss)