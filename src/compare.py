"""
Model comparison script - Compare custom vs official SegNet
"""

import torch
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.segnet_model import SegNet
from src.dataset import CamVidDataset, create_dataloader
from src.utils import SegmentationMetrics


class ModelComparator:
    """Compare two segmentation models"""
    
    def __init__(self, model_path1, model_path2, device=None, num_classes=11):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load models
        print(f"Loading model 1 from {model_path1}...")
        self.model1 = self._load_model(model_path1)
        
        print(f"Loading model 2 from {model_path2}...")
        self.model2 = self._load_model(model_path2)
        
        # Metrics
        self.metrics1 = SegmentationMetrics(num_classes=num_classes)
        self.metrics2 = SegmentationMetrics(num_classes=num_classes)
    
    def _load_model(self, model_path):
        """Load model from checkpoint"""
        model = SegNet(num_classes=self.num_classes, in_channels=3, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def benchmark_inference_time(self, sample_batch, num_iterations=10):
        """Benchmark inference time for both models"""
        sample_batch = sample_batch.to(self.device)
        
        # Warm up
        with torch.no_grad():
            _ = self.model1(sample_batch)
            _ = self.model2(sample_batch)
        
        # Benchmark model 1
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model1(sample_batch)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time1 = (time.time() - start) / num_iterations
        
        # Benchmark model 2
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model2(sample_batch)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time2 = (time.time() - start) / num_iterations
        
        return time1 * 1000, time2 * 1000  # Convert to ms
    
    def compare_on_dataset(self, loader):
        """Evaluate both models on dataset"""
        self.metrics1.reset()
        self.metrics2.reset()
        
        inference_times1 = []
        inference_times2 = []
        
        progress_bar = tqdm(loader, desc="Comparing models")
        
        with torch.no_grad():
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Model 1
                start = time.time()
                output1 = self.model1(images)
                time1 = (time.time() - start) * 1000  # ms
                inference_times1.append(time1)
                self.metrics1.update(output1, labels)
                
                # Model 2
                start = time.time()
                output2 = self.model2(images)
                time2 = (time.time() - start) * 1000  # ms
                inference_times2.append(time2)
                self.metrics2.update(output2, labels)
        
        # Compute metrics
        metrics1 = self.metrics1.get_metrics_dict()
        metrics2 = self.metrics2.get_metrics_dict()
        
        avg_time1 = np.mean(inference_times1)
        avg_time2 = np.mean(inference_times2)
        
        return {
            'model1': {
                'metrics': metrics1,
                'avg_inference_time_ms': avg_time1,
                'inference_times': inference_times1
            },
            'model2': {
                'metrics': metrics2,
                'avg_inference_time_ms': avg_time2,
                'inference_times': inference_times2
            }
        }
    
    def generate_comparison_report(self, results, class_names=None):
        """Generate detailed comparison report"""
        metrics1 = results['model1']['metrics']
        metrics2 = results['model2']['metrics']
        time1 = results['model1']['avg_inference_time_ms']
        time2 = results['model2']['avg_inference_time_ms']
        
        print("\n" + "="*80)
        print("MODEL COMPARISON REPORT")
        print("="*80 + "\n")
        
        # Overall metrics
        print(f"{'Metric':<30} {'Model 1':<20} {'Model 2':<20}")
        print("-"*70)
        
        metrics_to_compare = [
            ('Global Accuracy (%)', 'global_accuracy'),
            ('Class Avg Accuracy (%)', 'mean_class_accuracy'),
            ('Mean IoU (%)', 'mean_iou'),
            ('Inference Time (ms)', None)
        ]
        
        for display_name, metric_key in metrics_to_compare:
            if metric_key is None:
                print(f"{display_name:<30} {time1:<20.2f} {time2:<20.2f}")
            else:
                val1 = metrics1[metric_key]
                val2 = metrics2[metric_key]
                print(f"{display_name:<30} {val1:<20.2f} {val2:<20.2f}")
        
        # Per-class metrics
        if class_names:
            print("\n" + "="*80)
            print("PER-CLASS COMPARISON (IoU)")
            print("="*80)
            print(f"{'Class':<20} {'Model 1':<20} {'Model 2':<20}")
            print("-"*60)
            
            ious1 = metrics1['ious']
            ious2 = metrics2['ious']
            
            for i, class_name in enumerate(class_names):
                print(f"{class_name:<20} {ious1[i]:<20.2f} {ious2[i]:<20.2f}")
        
        print("\n" + "="*80)
    
    def save_comparison_results(self, results, output_dir):
        """Save comparison results to JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for JSON (remove numpy arrays)
        results_json = {
            'model1': {
                'metrics': {
                    'global_accuracy': float(results['model1']['metrics']['global_accuracy']),
                    'mean_class_accuracy': float(results['model1']['metrics']['mean_class_accuracy']),
                    'mean_iou': float(results['model1']['metrics']['mean_iou']),
                    'class_accuracies': results['model1']['metrics']['class_accuracies'].tolist(),
                    'ious': results['model1']['metrics']['ious'].tolist(),
                },
                'avg_inference_time_ms': float(results['model1']['avg_inference_time_ms'])
            },
            'model2': {
                'metrics': {
                    'global_accuracy': float(results['model2']['metrics']['global_accuracy']),
                    'mean_class_accuracy': float(results['model2']['metrics']['mean_class_accuracy']),
                    'mean_iou': float(results['model2']['metrics']['mean_iou']),
                    'class_accuracies': results['model2']['metrics']['class_accuracies'].tolist(),
                    'ious': results['model2']['metrics']['ious'].tolist(),
                },
                'avg_inference_time_ms': float(results['model2']['avg_inference_time_ms'])
            }
        }
        
        output_file = os.path.join(output_dir, 'comparison_results.json')
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        print(f"\n✅ Results saved to {output_file}")


def main():
    """Main comparison script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare SegNet models')
    parser.add_argument('--model1_path', type=str, required=True,
                       help='Path to first model')
    parser.add_argument('--model2_path', type=str, required=True,
                       help='Path to second model')
    parser.add_argument('--dataset_path', type=str, default='data/CamVid/',
                       help='Path to dataset')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='results/comparison/',
                       help='Directory to save comparison results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Create dataset
    dataset = CamVidDataset(args.dataset_path, split=args.split)
    loader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Compare models
    comparator = ModelComparator(args.model1_path, args.model2_path, device=device)
    results = comparator.compare_on_dataset(loader)
    
    # Generate report
    comparator.generate_comparison_report(results, class_names=CamVidDataset.CLASS_NAMES)
    
    # Save results
    comparator.save_comparison_results(results, args.output_dir)


if __name__ == "__main__":
    main()
