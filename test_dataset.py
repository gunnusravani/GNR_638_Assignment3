#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import CamVidDataset

print("Testing dataset loading...")
train_ds = CamVidDataset('data/CamVid', split='train')
print(f'Train dataset size: {len(train_ds)}')

val_ds = CamVidDataset('data/CamVid', split='val')
print(f'Val dataset size: {len(val_ds)}')

test_ds = CamVidDataset('data/CamVid', split='test')
print(f'Test dataset size: {len(test_ds)}')

print("\nDataset loading successful! ✅")
