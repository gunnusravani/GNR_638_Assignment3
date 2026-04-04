#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.segnet_model import SegNet

print("Testing SegNet model...")
model = SegNet(num_classes=11)
x = torch.randn(1, 3, 360, 480)
out = model(x)
print(f'✓ Input shape: {x.shape}')
print(f'✓ Output shape: {out.shape}')
print(f'✓ Model params: {sum(p.numel() for p in model.parameters()):,}')
print("\nModel test passed! Ready for training. ✅")
