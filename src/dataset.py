"""
Dataset handling for SegNet
Supports CamVid and general semantic segmentation datasets
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random


class ToySegmentationDataset(Dataset):
    """Synthetic toy dataset for quick testing and validation"""
    
    def __init__(self, num_samples=100, img_size=(360, 480), num_classes=11, 
                 split='train', transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.split = split
        self.transform = transform
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random image and label
        image = torch.randint(0, 256, (3, self.img_size[0], self.img_size[1])).float()
        image = image / 255.0
        
        # Generate simple segmentation mask (random regions)
        mask = torch.randint(0, self.num_classes, (self.img_size[0], self.img_size[1]))
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask.long()


class CamVidDataset(Dataset):
    """
    CamVid dataset loader
    Directory structure expected:
        data/CamVid/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
    """
    
    # CamVid class names and colors
    CLASS_NAMES = [
        'road', 'sidewalk', 'tree', 'car', 'fence', 'pedestrian',
        'building', 'pole', 'sky', 'bicycle', 'sign'
    ]
    
    # Colors for each class (RGB)
    CLASS_COLORS = {
        0: [128, 64, 128],      # road
        1: [244, 35, 232],      # sidewalk
        2: [107, 142, 35],      # tree
        3: [70, 70, 70],        # car
        4: [100, 100, 40],      # fence
        5: [204, 5, 255],       # pedestrian
        6: [230, 0, 0],         # building
        7: [110, 110, 110],     # pole
        8: [70, 130, 180],      # sky
        9: [119, 11, 32],       # bicycle
        10: [0, 0, 142],        # sign
    }
    
    def __init__(self, root_dir, split='train', transform=None, img_size=(360, 480)):
        """
        Args:
            root_dir: Root directory of dataset
            split: 'train', 'val', or 'test'
            transform: Preprocessing transforms
            img_size: Target image size (height, width)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.num_classes = len(self.CLASS_NAMES)
        
        # Paths
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.label_dir = os.path.join(root_dir, split, 'labels')
        
        # Get image files
        if os.path.exists(self.img_dir):
            self.img_files = sorted([f for f in os.listdir(self.img_dir) 
                                    if f.endswith(('.png', '.jpg', '.jpeg'))])
        else:
            self.img_files = []
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load label (convert RGB label image to class index)
        label_name = img_name.replace('.png', '_L.png')
        label_path = os.path.join(self.label_dir, label_name)
        
        if os.path.exists(label_path):
            label_img = Image.open(label_path).convert('RGB')
            label = self._rgb_to_class(label_img)
        else:
            # If label doesn't exist, create dummy label
            label = Image.new('L', image.size, 255)
            label = np.array(label)
        
        # Resize
        image = image.resize(self.img_size, Image.BILINEAR)
        if isinstance(label, Image.Image):
            label = label.resize(self.img_size, Image.NEAREST)
            label = np.array(label)
        else:
            label = Image.fromarray(label.astype(np.uint8)).resize(self.img_size, Image.NEAREST)
            label = np.array(label)
        
        # Convert to tensors
        image = transforms.ToTensor()(image)
        label = torch.from_numpy(label)
        
        # Apply transforms if any
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label.long()
    
    @staticmethod
    def _rgb_to_class(label_img):
        """Convert RGB label image to class indices"""
        label_array = np.array(label_img)
        class_map = np.zeros((label_array.shape[0], label_array.shape[1]), dtype=np.uint8)
        
        # Map RGB colors to class indices
        colors_list = list(CamVidDataset.CLASS_COLORS.values())
        for class_idx, color in enumerate(colors_list):
            # Find pixels matching this color
            mask = np.all(label_array == color, axis=2)
            class_map[mask] = class_idx
        
        return class_map


class SegmentationTransform:
    """Augmentation transforms for training"""
    
    def __init__(self, img_size=(360, 480), augment=True):
        self.img_size = img_size
        self.augment = augment
    
    def __call__(self, image, mask):
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
            
            # Random vertical flip
            if random.random() > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.randint(-10, 10)
                image = F.rotate(image, angle)
                mask = F.rotate(mask, angle)
            
            # Random brightness/contrast
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                image = F.adjust_brightness(image, brightness_factor)
        
        # Normalize
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
        
        return image, mask


def create_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0):
    """Create DataLoader"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                     num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    print("Testing dataset loading...")
    
    # Test toy dataset
    toy_dataset = ToySegmentationDataset(num_samples=10, num_classes=11)
    print(f"Toy dataset size: {len(toy_dataset)}")
    
    img, mask = toy_dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Test dataloader
    loader = create_dataloader(toy_dataset, batch_size=2)
    batch_img, batch_mask = next(iter(loader))
    print(f"Batch image shape: {batch_img.shape}")
    print(f"Batch mask shape: {batch_mask.shape}")
    
    print("Dataset test successful!")
