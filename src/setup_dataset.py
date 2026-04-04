"""
Dataset setup script - Download and prepare CamVid dataset
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path


class CamVidDatasetDownloader:
    """Download and prepare CamVid dataset"""
    
    DATASET_URL = "http://mi.eng.cam.ac.uk/research/projects/VideoSegmentation/data/CamVidSeq1.zip"
    
    def __init__(self, root_dir='data/CamVid'):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self):
        """Download CamVid dataset"""
        print("Note: CamVid dataset must be downloaded manually from:")
        print("http://mi.eng.cam.ac.uk/research/projects/VideoSegmentation/data.html")
        print("\nExpected directory structure:")
        print("""
        data/CamVid/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
        """)
        print("\nCreate directories and place images manually.")
    
    def create_dummy_dataset(self, num_train=5, num_val=2, num_test=2):
        """Create dummy dataset for testing"""
        import cv2
        import numpy as np
        
        print("Creating dummy CamVid dataset for testing...")
        
        splits = {
            'train': num_train,
            'val': num_val,
            'test': num_test
        }
        
        for split, count in splits.items():
            img_dir = self.root_dir / split / 'images'
            label_dir = self.root_dir / split / 'labels'
            
            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                # Create dummy image
                img = np.random.randint(0, 256, (360, 480, 3), dtype=np.uint8)
                img_path = img_dir / f'{split}_image_{i:03d}.png'
                cv2.imwrite(str(img_path), img)
                
                # Create dummy label
                label = np.random.randint(0, 11, (360, 480), dtype=np.uint8)
                # Convert to RGB for label
                label_rgb = np.zeros((360, 480, 3), dtype=np.uint8)
                label_rgb[:, :, 0] = label * 23  # Spread across channels
                label_path = label_dir / f'{split}_image_{i:03d}_L.png'
                cv2.imwrite(str(label_path), label_rgb)
        
        print(f"✅ Dummy dataset created at {self.root_dir}")
        print(f"   Train: {num_train} images")
        print(f"   Val: {num_val} images")
        print(f"   Test: {num_test} images")


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup CamVid dataset')
    parser.add_argument('--dummy', action='store_true', 
                       help='Create dummy dataset for testing')
    parser.add_argument('--num_train', type=int, default=5,
                       help='Number of training samples for dummy dataset')
    parser.add_argument('--num_val', type=int, default=2,
                       help='Number of validation samples for dummy dataset')
    parser.add_argument('--num_test', type=int, default=2,
                       help='Number of test samples for dummy dataset')
    parser.add_argument('--root_dir', type=str, default='data/CamVid',
                       help='Root directory for dataset')
    
    args = parser.parse_args()
    
    downloader = CamVidDatasetDownloader(root_dir=args.root_dir)
    
    if args.dummy:
        downloader.create_dummy_dataset(
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test
        )
    else:
        downloader.download()


if __name__ == "__main__":
    main()
