"""
Dataset verification script
Check if CamVid dataset is properly formatted and complete
"""

import os
from PIL import Image
import numpy as np


def verify_dataset(root_dir='data/CamVid'):
    """Verify CamVid dataset structure and integrity"""
    
    print("=" * 70)
    print("CamVid Dataset Verification Report")
    print("=" * 70)
    
    splits = ['train', 'val', 'test']
    summary = {}
    
    all_valid = True
    
    for split in splits:
        print(f"\n📂 Checking {split.upper()} split...")
        
        img_dir = os.path.join(root_dir, split)
        label_dir = os.path.join(root_dir, f"{split}annot")
        
        if not os.path.exists(img_dir):
            print(f"  ✗ Image directory not found: {img_dir}")
            all_valid = False
            continue
        
        if not os.path.exists(label_dir):
            print(f"  ✗ Label directory not found: {label_dir}")
            all_valid = False
            continue
        
        # Get files
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
        
        print(f"  ✓ Image directory found: {len(img_files)} images")
        print(f"  ✓ Label directory found: {len(label_files)} labels")
        
        # Check counts match
        if len(img_files) != len(label_files):
            print(f"  ✗ Image and label counts don't match!")
            all_valid = False
        else:
            print(f"  ✓ Image-label count match: {len(img_files)}")
        
        # Sample check
        if img_files and label_files:
            sample_img_path = os.path.join(img_dir, img_files[0])
            sample_label_path = os.path.join(label_dir, label_files[0])
            
            try:
                img = Image.open(sample_img_path)
                label = Image.open(sample_label_path)
                
                print(f"\n  Sample Image Properties:")
                print(f"    File: {img_files[0]}")
                print(f"    Size: {img.size[0]}×{img.size[1]} pixels")
                print(f"    Mode: {img.mode}")
                print(f"    Format: {img.format}")
                
                print(f"\n  Sample Label Properties:")
                print(f"    File: {label_files[0]}")
                print(f"    Size: {label.size[0]}×{label.size[1]} pixels")
                print(f"    Mode: {label.mode}")
                print(f"    Format: {label.format}")
                
                # Check size match
                if img.size == label.size:
                    print(f"\n  ✓ Sample image and label have matching dimensions")
                else:
                    print(f"\n  ✗ Sample image and label have different dimensions!")
                    all_valid = False
                
                # Get unique classes in sample label
                label_array = np.array(label)
                if label.mode == 'RGB':
                    unique_colors = len(np.unique(label_array.reshape(-1, 3), axis=0))
                    print(f"  ✓ Sample label has {unique_colors} unique color(s)")
                else:
                    unique_classes = len(np.unique(label_array))
                    print(f"  ✓ Sample label has {unique_classes} unique class(es)")
                
            except Exception as e:
                print(f"  ✗ Error reading sample files: {e}")
                all_valid = False
        
        summary[split] = {
            'images': len(img_files),
            'labels': len(label_files),
            'valid': len(img_files) == len(label_files)
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    total_images = sum(s['images'] for s in summary.values())
    total_labels = sum(s['labels'] for s in summary.values())
    
    for split, stats in summary.items():
        status = "✓" if stats['valid'] else "✗"
        print(f"{status} {split.upper():8} - {stats['images']:4} images, {stats['labels']:4} labels")
    
    print(f"\nTotal: {total_images} images, {total_labels} labels")
    
    if all_valid and total_images == total_labels > 0:
        print("\n✅ Dataset verification PASSED - Data is in correct format!")
        return True
    else:
        print("\n❌ Dataset verification FAILED - Please check the data!")
        return False


if __name__ == "__main__":
    verify_dataset()
