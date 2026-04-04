#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.dataset import CamVidDataset

# Load a test sample
dataset = CamVidDataset('data/CamVid', split='test')
image, label = dataset[0]

print("Sample from test set:")
print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")
print(f"Label unique classes: {np.unique(label.numpy())}")
print(f"Label class distribution:")
unique, counts = np.unique(label.numpy(), return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls} ({CamVidDataset.CLASS_NAMES[cls]}): {count} pixels ({100*count/label.numel():.1f}%)")

# Check the actual colors in a label image
from PIL import Image
import os

label_dir = 'data/CamVid/test'  # Look for actual label files
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])

if label_files:
    sample_label_path = os.path.join(label_dir, label_files[0])
    img = Image.open(sample_label_path).convert('RGB')
    img_array = np.array(img)
    
    print(f"\nActual colors in first label image ({label_files[0]}):")
    unique_colors = np.unique(img_array.reshape(-1, 3), axis=0)
    print(f"Found {len(unique_colors)} unique colors in image")
    print("First 5 unique colors:")
    for color in unique_colors[:5]:
        print(f"  RGB{tuple(color)}")
