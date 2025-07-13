import os
from pathlib import Path
from typing import Optional, Callable, Union, Sequence,List
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset,DataLoader
import zipfile
from PIL import Image
from torchvision import transforms
from pytorch_lightning import LightningDataModule

class AnimeDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        imgs = sorted(list(self.root_dir.glob("*.jpg")))
        self.image_paths = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
        self.transform = transform

    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = default_loader(image_path)
        if self.transform:
            image = self.transform(image)
        
        return image, 0.0
    
    

class PadToSquare():
    def __call__(self,image: Image.Image) -> Image.Image:
        w, h  = image.size
        max_wh = max(w, h)
        pad_w = (max_wh - w) // 2
        pad_h = (max_wh - h) // 2
        padding = (pad_w, pad_h, max_wh - w - pad_w, max_wh - h - pad_h)
        return transforms.functional.pad(image,padding,fill=0, padding_mode='constant')
    
    
class VAEDataset(LightningDataModule):

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def setup(self, stage: Optional[str] = None) -> None:
    
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            PadToSquare(), 
            transforms.Resize((64, 64)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        val_transforms = transforms.Compose([
            PadToSquare(), 
            transforms.Resize((64, 64)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
            
        self.train_dataset = AnimeDataset(
            self.data_dir,
            split='train',
            transform=train_transforms
        )
        
        self.val_dataset = AnimeDataset(
            self.data_dir,
            split='val',
            transform=val_transforms
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
        
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )