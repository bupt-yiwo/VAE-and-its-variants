from typing import List, Any
import torch
from torch import nn
from abc import abstractmethod

class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError
    
    def decode(self, input: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError
    
    def sample(self, batch_size: int, current_device: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def generate(self, x:torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, x:torch.Tensor, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def loss_fuction(self, *inputs: Any, **kwargs) -> torch.Tensor:
        pass