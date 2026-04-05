#!/usr/bin/env python3
"""Extract actual class colors from official SegNet-Tutorial CamVid labels"""

import numpy as np
from PIL import Image
import os
from collections import Counter

# Check a few label images from the official repo
label_dir = 'SegNet-Tutorial/CamVid/train'
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])[:5]

print("Analyzing official SegNet-Tutorial label images...")
print("="*70)

all_colors = []

for fname in label_files:
    label_path = os.path.join(label_dir, fname)
    print(f"\nAnalyzing: {fname}")
    
    img = Image.open(label_path).convert('RGB')
    img_array = np.array(img, dtype=np.uint8)
    
    unique_colors, counts = np.unique(img_array.reshape(-1, 3), axis=0, return_counts=True)
    
    print(f"  Unique colors: {len(unique_colors)}")
    print(f"  Top 5 colors:")
    
    sorted_idx = np.argsort(-counts)
    for i, idx in enumerate(sorted_idx[:5]):
        color = tuple(unique_colors[idx])
        count = counts[idx]
        pct = 100 * count / img_array.size
        print(f"    {i+1}. RGB{color} - {count:6d} pixels ({pct:5.2f}%)")
        all_colors.append(color)

print("\n" + "="*70)
print("All unique colors found across sampled label images:")
print("="*70)

color_counts = Counter(all_colors)
for i, (color, count) in enumerate(color_counts.most_common(15)):
    pct = 100 * count / len(all_colors) if all_colors else 0
    print(f"{i+1:2d}. RGB{str(color):30s} - {count:3d} occurrences ({pct:5.1f}%)")
