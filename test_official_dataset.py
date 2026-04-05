#!/usr/bin/env python3
"""Test dataset loading with official SegNet-Tutorial CamVid (grayscale indexed encoding)"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import CamVidDataset
import numpy as np

print("Testing dataset loading with official SegNet-Tutorial CamVid...")
print("="*70)

dataset = CamVidDataset('data/CamVid', split='train')
image, label = dataset[0]

print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")
print(f"Unique labels: {np.unique(label.numpy())}")
print(f"\nLabel distribution:")
unique, counts = np.unique(label.numpy(), return_counts=True)
class_names = ['road', 'sidewalk', 'tree', 'car', 'fence', 'pedestrian', 'building', 'pole', 'sky', 'bicycle', 'sign']

for cls, count in zip(unique, counts):
    if cls < 11:
        pct = 100 * count / label.numel()
        print(f"  Class {cls:2d} ({class_names[cls]:12s}): {count:7d} pixels ({pct:5.2f}%)")

print(f"\n{'='*70}")
print("✓ Dataset loading successful!")
