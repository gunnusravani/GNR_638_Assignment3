#!/usr/bin/env python3
"""Analyze dominant colors in CamVid dataset labels"""

import numpy as np
from PIL import Image
import os
from collections import Counter

label_dir = 'data/CamVid/test'
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])[:5]

all_colors = []
print("Analyzing first 5 test images...")
print(f"Label directory: {label_dir}")
print(f"Found {len(label_files)} images to analyze\n")

for fname in label_files:
    label_path = os.path.join(label_dir, fname)
    print(f"Processing: {fname}")
    img = Image.open(label_path).convert('RGB')
    img_array = np.array(img, dtype=np.uint8)
    
    # Get unique colors
    pixels = img_array.reshape(-1, 3)
    for pixel in pixels:
        all_colors.append(tuple(pixel))

# Get most common colors
color_counts = Counter(all_colors)
print(f"\n{'='*70}")
print(f"Top 20 most common colors across test set:")
print(f"{'='*70}")
for i, (color, count) in enumerate(color_counts.most_common(20)):
    pct = 100 * count / len(all_colors)
    print(f"{i+1:2d}. RGB{str(color):30s} - {count:8d} pixels ({pct:5.2f}%)")

print(f"\n{'='*70}")
print(f"Expected class colors (from CamVid paper):")
print(f"{'='*70}")

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

for cls, color in sorted(CLASS_COLORS.items()):
    class_names = ['road', 'sidewalk', 'tree', 'car', 'fence', 'pedestrian',
                   'building', 'pole', 'sky', 'bicycle', 'sign']
    print(f"Class {cls:2d} ({class_names[cls]:12s}): RGB{tuple(color)}")
