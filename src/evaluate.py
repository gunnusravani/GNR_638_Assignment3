"""
Evaluation and inference script for SegNet
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segnet_model import SegNet
from src.dataset import CamVidDataset, ToySegmentationDataset, create_dataloader
from src.utils import SegmentationMetrics


class SegNetEvaluator:
    """Evaluator class for SegNet"""
    
    def __init__(self, model_path, num_classes=11, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = SegNet(num_classes=num_classes, in_channels=3, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.num_classes = num_classes
        print(f"Loaded model from {model_path}")
    
    def predict(self, image):
        """
        Predict segmentation for a single image
        
        Args:
            image: Tensor of shape (3, H, W) or (1, 3, H, W)
        
        Returns:
            prediction: Tensor of shape (H, W) with class indices
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
        
        prediction = torch.argmax(output, dim=1).squeeze(0)
        return prediction.cpu()
    
    def predict_batch(self, loader):
        """Predict on batch of images, returning raw logits"""
        predictions = []
        targets = []
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Predicting"):
                images = images.to(self.device)
                outputs = self.model(images)  # (B, C, H, W) - raw logits
                
                predictions.append(outputs.cpu())
                targets.append(labels.cpu())
        
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        
        return predictions, targets
    
    def evaluate(self, loader, class_names=None):
        """Evaluate on test set"""
        logits, targets = self.predict_batch(loader)
        
        # Compute metrics
        metrics = SegmentationMetrics(num_classes=self.num_classes)
        
        # Pass raw logits to metrics (they handle argmax internally)
        metrics.update(logits, targets)
        
        # Print metrics
        metrics.print_metrics(class_names=class_names)
        
        return metrics.get_metrics_dict()


def visualize_predictions(image, prediction, target, save_path=None):
    """Visualize prediction vs ground truth"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image
    if image.shape[0] == 3:
        img_display = image.permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
    else:
        img_display = image.numpy()
    
    axes[0].imshow(img_display)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(prediction.numpy(), cmap='tab20')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Ground truth
    axes[2].imshow(target.numpy(), cmap='tab20')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()


def compare_models(custom_model_path, official_model_path, test_loader, num_classes=11):
    """Compare custom vs official SegNet"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate custom model
    print("\n" + "="*60)
    print("EVALUATING CUSTOM SEGNET")
    print("="*60)
    custom_evaluator = SegNetEvaluator(custom_model_path, num_classes=num_classes, device=device)
    custom_metrics = custom_evaluator.evaluate(test_loader)
    
    # Evaluate official model
    print("\n" + "="*60)
    print("EVALUATING OFFICIAL SEGNET")
    print("="*60)
    official_evaluator = SegNetEvaluator(official_model_path, num_classes=num_classes, device=device)
    official_metrics = official_evaluator.evaluate(test_loader)
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    comparison = {
        'metric': [],
        'custom': [],
        'official': [],
        'difference': []
    }
    
    metrics_to_compare = ['global_accuracy', 'mean_class_accuracy', 'mean_iou']
    
    for metric in metrics_to_compare:
        custom_val = custom_metrics[metric]
        official_val = official_metrics[metric]
        diff = custom_val - official_val
        
        comparison['metric'].append(metric)
        comparison['custom'].append(custom_val)
        comparison['official'].append(official_val)
        comparison['difference'].append(diff)
        
        print(f"{metric:<25} Custom: {custom_val:8.2f}  Official: {official_val:8.2f}  Diff: {diff:+.2f}")
    
    return {
        'custom': custom_metrics,
        'official': official_metrics,
        'comparison': comparison
    }


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate SegNet')
    parser.add_argument('--model_path', type=str, default='./models/custom_segnet/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='toy',
                       choices=['toy', 'camvid'],
                       help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    if args.dataset == 'toy':
        dataset = ToySegmentationDataset(num_samples=20, num_classes=11, split=args.split)
    else:
        dataset = CamVidDataset('data/CamVid', split=args.split)
    
    loader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate
    evaluator = SegNetEvaluator(args.model_path, num_classes=11, device=device)
    metrics = evaluator.evaluate(loader, class_names=CamVidDataset.CLASS_NAMES)
    
    # Save results
    os.makedirs('./results', exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        else:
            metrics_serializable[key] = value
    
    with open('./results/evaluation_results.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    
    print(f"\nResults saved to ./results/evaluation_results.json")


if __name__ == "__main__":
    main()
