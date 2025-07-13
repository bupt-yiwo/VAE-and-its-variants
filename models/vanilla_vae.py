import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List, Any

class VanillaVAE(BaseVAE):
    
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hiddendims: List = None,
                 **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        
        modules = []
        if hiddendims is None:
            hiddendims = [32, 64, 128, 258, 512]
        
        for hiddendim in hiddendims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels = hiddendim,
                        kernel_size = 3 , stride =2 , padding = 1),
                    nn.BatchNorm2d(hiddendim),
                    nn.LeakyReLU())
                )
            in_channels = hiddendim
            
        self.encoder = nn.Sequential(*modules)
        
        self.fc_mu = nn.Linear(hiddendims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hiddendims[-1] * 4, latent_dim)
        
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hiddendims[-1] * 4)
        hiddendims.reverse()
        
        for i in range(len(hiddendims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hiddendims[i], out_channels = hiddendims[i+1],
                        kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                    nn.BatchNorm2d(hiddendims[i+1]),
                    nn.LeakyReLU()
                    )
                )
            
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hiddendims[-1], out_channels= hiddendims[-1],
                kernel_size= 3, stride = 2, padding= 1, output_padding=1
            ),
            nn.BatchNorm2d(hiddendims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                hiddendims[-1], out_channels= 3,
                kernel_size= 3, padding= 1
            ),
            nn.Tanh()
        )
        
    def encode(self, x: torch.Tensor) -> Any:
        result = self.encoder(x)
        result = result.view(result.size(0), -1)
        mu =  self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(result.size(0), -1, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result 
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return eps * std + mu
    
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return [self.decode(z), input, mu, logvar]
    
    def loss_function(self, *args, **kwargs) -> dict:
        recon = args[0]
        target = args[1]
        mu, logvar = args[2], args[3]
        
        
        recon_loss = F.mse_loss(recon,target)
        
        kld_weight = kwargs["M_N"]
        kld_loss = torch.mean(-0.5 * torch.sum(1 +  logvar - mu.pow(2) - logvar.exp(), dim= 1),dim = 0)
        
        loss = recon_loss + kld_weight * kld_loss
        
        return {'loss': loss, 'Reconstruction_Loss':recon_loss.detach(), 'KLD':-kld_loss.detach()}
    
    def sample(self, num_samples:int, current_device: int, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples,self.latent_dim)
        z = z.to(current_device)
        
        samles = self.decode(z)
        return samles
    
    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]