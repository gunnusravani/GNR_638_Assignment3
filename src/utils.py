"""
Utility functions for evaluation metrics and loss computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix


class SegNetLoss(nn.Module):
    """Weighted Cross-Entropy Loss with optional class balancing"""
    
    def __init__(self, num_classes=11, weight=None, ignore_index=255):
        super(SegNetLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='mean')
    
    def forward(self, outputs, targets, weight_map=None):
        """
        Args:
            outputs: Model output (B, C, H, W)
            targets: Ground truth labels (B, H, W)
            weight_map: Optional per-pixel weight map (B, H, W)
        """
        loss = self.criterion(outputs, targets)
        return loss


def compute_class_weights_median_frequency(target, num_classes, ignore_index=255):
    """
    Compute class weights using median frequency balancing
    
    Args:
        target: Tensor of shape (B, H, W) containing class labels
        num_classes: Number of classes
        ignore_index: Index to ignore
    
    Returns:
        weights: Tensor of shape (num_classes,)
    """
    flat_target = target.view(-1)
    
    # Compute class frequencies
    class_counts = torch.zeros(num_classes, device=target.device)
    for c in range(num_classes):
        if c != ignore_index:
            class_counts[c] = (flat_target == c).sum().float()
    
    # Get median frequency
    non_zero_counts = class_counts[class_counts > 0]
    median_freq = torch.median(non_zero_counts)
    
    # Compute weights
    weights = median_freq / (class_counts + 1e-10)
    weights[class_counts == 0] = 0
    
    return weights


class SegmentationMetrics:
    """Compute segmentation metrics: accuracy, IoU, etc."""
    
    def __init__(self, num_classes=11, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, predictions, targets):
        """
        Update confusion matrix
        
        Args:
            predictions: Model predictions (B, C, H, W) - logits or probabilities
            targets: Ground truth labels (B, H, W)
        """
        # Get class predictions
        if predictions.dim() == 4:
            preds = torch.argmax(predictions, dim=1)  # (B, H, W)
        else:
            preds = predictions
        
        # Flatten
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # Filter ignore_index
        valid_mask = targets != self.ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        # Update confusion matrix
        cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
        self.confusion_matrix += cm
    
    def get_global_accuracy(self):
        """Global accuracy: correct predictions / total predictions"""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return (correct / total) * 100 if total > 0 else 0
    
    def get_class_accuracy(self):
        """Per-class accuracy"""
        class_accuracies = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp_fn = self.confusion_matrix[i, :].sum() - tp
            acc = (tp / (tp + fp_fn)) * 100 if (tp + fp_fn) > 0 else 0
            class_accuracies.append(acc)
        return np.array(class_accuracies)
    
    def get_mean_class_accuracy(self):
        """Mean of per-class accuracies"""
        return np.mean(self.get_class_accuracy())
    
    def get_iou(self):
        """Intersection over Union (Jaccard Index) per class"""
        ious = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            iou = (tp / (tp + fp + fn)) * 100 if (tp + fp + fn) > 0 else 0
            ious.append(iou)
        return np.array(ious)
    
    def get_mean_iou(self):
        """Mean Intersection over Union"""
        return np.mean(self.get_iou())
    
    def get_metrics_dict(self):
        """Return all metrics as dictionary"""
        return {
            'global_accuracy': self.get_global_accuracy(),
            'mean_class_accuracy': self.get_mean_class_accuracy(),
            'mean_iou': self.get_mean_iou(),
            'class_accuracies': self.get_class_accuracy(),
            'ious': self.get_iou(),
        }
    
    def print_metrics(self, class_names=None):
        """Print metrics in readable format"""
        metrics = self.get_metrics_dict()
        
        print(f"\n{'='*60}")
        print(f"Global Accuracy: {metrics['global_accuracy']:.2f}%")
        print(f"Mean Class Accuracy: {metrics['mean_class_accuracy']:.2f}%")
        print(f"Mean IoU: {metrics['mean_iou']:.2f}%")
        print(f"{'='*60}")
        
        if class_names is not None:
            print(f"\n{'Class':<20} {'Accuracy':<15} {'IoU':<15}")
            print(f"{'-'*50}")
            for i, name in enumerate(class_names):
                acc = metrics['class_accuracies'][i]
                iou = metrics['ious'][i]
                print(f"{name:<20} {acc:<14.2f}% {iou:<14.2f}%")


if __name__ == "__main__":
    # Test metrics computation
    print("Testing SegmentationMetrics...")
    
    metrics = SegmentationMetrics(num_classes=11)
    
    # Create dummy predictions and targets
    batch_size, height, width = 2, 256, 256
    preds = torch.randint(0, 11, (batch_size, height, width))
    targets = torch.randint(0, 11, (batch_size, height, width))
    
    # Create logits format
    logits = torch.randn(batch_size, 11, height, width)
    
    metrics.update(logits, targets)
    metrics.print_metrics()
    
    print("Metrics test successful!")
