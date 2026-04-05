"""Debug script to check test label format"""
from PIL import Image
import numpy as np
import os

# Check a validation label
val_label = 'data/CamVid/valannot/0016E5_07959.png'
test_label = 'data/CamVid/testannot/0001TP_008670.png'

print('=' * 60)
print('VALIDATION LABEL')
print('=' * 60)
val_img = Image.open(val_label)
print(f'Mode: {val_img.mode}')
val_arr = np.array(val_img)
print(f'Shape: {val_arr.shape}')
print(f'Dtype: {val_arr.dtype}')
print(f'Min-Max: {val_arr.min()}-{val_arr.max()}')
print(f'Unique values: {np.unique(val_arr)}')
print()

print('=' * 60)
print('TEST LABEL')
print('=' * 60)
test_img = Image.open(test_label)
print(f'Mode: {test_img.mode}')
test_arr = np.array(test_img)
print(f'Shape: {test_arr.shape}')
print(f'Dtype: {test_arr.dtype}')

if len(test_arr.shape) == 3:
    print(f'\n⚠️ TEST LABEL HAS 3 CHANNELS (RGB)!')
    print(f'Unique R channel values: {len(np.unique(test_arr[:,:,0]))}')
    print(f'Unique G channel values: {len(np.unique(test_arr[:,:,1]))}')
    print(f'Unique B channel values: {len(np.unique(test_arr[:,:,2]))}')
    print(f'Sample RGB triplets: {np.unique(test_arr.reshape(-1, 3), axis=0)[:10]}')
else:
    print(f'Min-Max: {test_arr.min()}-{test_arr.max()}')
    print(f'Unique values: {np.unique(test_arr)}')
