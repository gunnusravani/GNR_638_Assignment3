"""Discover the correct CamVid class mapping"""
import numpy as np
from PIL import Image
import os
from collections import Counter

# Sample from validation set to discover class mapping
val_path = '/Users/sravani/Documents/VSCode_projects/GNR_638_Assignment3/SegNet-Tutorial/CamVid/valannot'
val_files = sorted(os.listdir(val_path))[:10]  # First 10 files

all_classes = Counter()

for fname in val_files:
    img = Image.open(os.path.join(val_path, fname))
    arr = np.array(img)
    unique_in_file = np.unique(arr)
    for cls in unique_in_file:
        all_classes[int(cls)] += 1
    print(f"{fname}: classes {unique_in_file}")

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION IN VALIDATION SET")
print("=" * 60)
for cls in sorted(all_classes.keys()):
    print(f"Class {cls:2d}: {all_classes[cls]:6d} pixels")

print("\nTotal unique classes:", len(all_classes))
print("Class indices range: 0-", max(all_classes.keys()))
