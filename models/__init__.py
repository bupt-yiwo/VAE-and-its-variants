from .base import *
from .vanilla_vae import *
from .vq_vae import *
# from .vq_vae import *


VAE = VanillaVAE

vae_models = {'VanillaVAE':VanillaVAE,
              'VQ_VAE':VQVAE}