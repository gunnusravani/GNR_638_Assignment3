#!/usr/bin/env python3
"""Debug script to check label colors across different data splits"""

import numpy as np
from PIL import Image
import os

splits = ['train', 'val', 'test']

for split in splits:
    label_dir = f'data/CamVid/{split}'
    if not os.path.exists(label_dir):
        label_dir = f'data/CamVid/{split}annot'
    
    if not os.path.exists(label_dir):
        print(f"Directory not found: {label_dir}")
        continue
    
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])[:1]
    
    if label_files:
        print(f"\n{'='*60}")
        print(f"Split: {split.upper()}")
        print(f"{'='*60}")
        
        label_path = os.path.join(label_dir, label_files[0])
        img = Image.open(label_path)
        print(f"Image mode: {img.mode}")
        print(f"Image size: {img.size}")
        
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb, dtype=np.uint8)
        
        unique_colors, counts = np.unique(img_array.reshape(-1, 3), axis=0, return_counts=True)
        print(f"Unique colors: {len(unique_colors)}")
        
        # Top 5 colors
        sorted_idx = np.argsort(-counts)
        print("Top 5 colors:")
        for i, idx in enumerate(sorted_idx[:5]):
            color = tuple(unique_colors[idx])
            count = counts[idx]
            pct = 100 * count / img_array.size
            print(f"  {i+1}. RGB{str(color):30s} - {count:7d} pixels ({pct:5.2f}%)")
