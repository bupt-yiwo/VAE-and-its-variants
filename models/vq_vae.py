import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
import math

class VectorQuantizer(nn.Module):
    def __init__(self, 
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25
                 ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        # self.embedding.weight.data.uniform_(-1.0/self.K , 1.0 / self.K)
        # nn.init.uniform_(self.embedding.weight, -1 / math.sqrt(self.D), 1 / math.sqrt(self.D))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        
    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = latents.permute(0,2,3,1).contiguous() # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)
        
        dist = torch.sum( flat_latents ** 2, dim = 1, keepdim= True) + \
               torch.sum(self.embedding.weight ** 2, dim= 1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())
               
        encoding_inds = torch.argmin(dist, dim= 1).unsqueeze(1)
        
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device = device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)
        
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized_latents = quantized_latents.view(latents_shape)
        
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        
        vq_loss = commitment_loss * self.beta + embedding_loss
        
        with torch.no_grad():
            unique_codes = torch.unique(encoding_inds)
            usage_ratio = len(unique_codes) / self.K
            print(f"[Codebook usage] {len(unique_codes)} / {self.K} codes used ({usage_ratio:.2%})")

        quantized_latents = latents + (quantized_latents - latents.detach())
        
        return quantized_latents.permute(0,3,1,2).contiguous(), vq_loss
    
class ResidualLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias= False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        )
    
    def forward(self,input: torch.Tensor) -> torch.Tensor:
        return input + self.resblock(input)
    

class VQVAE(BaseVAE):
    
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        num_embeddings: int,
        hiddendims: List = None,
        beta: float = 0.25,
        img_size: int = 64,
        ** kwargs
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        
        modules = []
        if hiddendims is None:
            hiddendims = [128, 256]
            
        for hiddendim in hiddendims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=hiddendim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            in_channels = hiddendim
            
        modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU()
                )
                )
        
        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        
        modules.append(nn.LeakyReLU())
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels=self.embedding_dim, kernel_size=1, stride=1),
                nn. LeakyReLU()
            )
        )
        
        self.encoder = nn.Sequential(*modules)
        self.vq_layer = VectorQuantizer(self.num_embeddings, self.embedding_dim, self.beta)
        
        
        modules = []
        
        modules.append(
            nn.Sequential(
                nn.Conv2d(self.embedding_dim, out_channels= hiddendims[-1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
        )
        for _ in range(6):   
            modules.append(ResidualLayer(hiddendims[-1],hiddendims[-1]))
        modules.append(nn.LeakyReLU())
        
        hiddendims.reverse()
        for i in range(len(hiddendims) - 1):
            modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddendims[i],
                                   hiddendims[i+1],
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),
                nn.LeakyReLU()
                            )
                
                            )
        
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(
                hiddendims[-1],
                out_channels = 3,
                kernel_size = 4,
                stride = 2, padding=1
            ),
            nn.Tanh())
        )
        
        self.decoder = nn.Sequential(*modules)
        
    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        result = self.encoder(input)
        result = F.normalize(result, dim=1)
        return [result]
    
    def decode(self, z:torch.Tensor) -> torch.Tensor:
        result = self.decoder(z)
        return result
    
    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        encoding = self.encode(input)
        quantized_inputs, vq_loss = self.vq_layer(encoding[0])
        return [self.decode(quantized_inputs), input, vq_loss]
    
    def loss_function(
        self,
        *args,
        **kwargs
    )-> dict:
        recons = args[0]
        target = args[1]
        vq_loss = args[2]
        
        recon_loss = F.mse_loss(recons, target)
        loss = 5 * recon_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recon_loss,
                'VQ_Loss':vq_loss}
    def sample(self, num_samples: int, current_device: Union[int, str], **kwargs) -> torch.Tensor:
        raise Warning("VQVAE sampler is not implemented.")
    
    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(x)[0]
    