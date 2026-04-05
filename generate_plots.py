#!/usr/bin/env python3
"""
Generate plots for training history and evaluation results
"""
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# Create results/plots directory
plots_dir = Path('results/plots')
plots_dir.mkdir(exist_ok=True)

# Load training history
with open('results/training_history.json', 'r') as f:
    history = json.load(f)

# Load evaluation results
with open('results/evaluation_results.json', 'r') as f:
    eval_results = json.load(f)

epochs = np.arange(1, len(history['train_loss']) + 1)

# ============================================================================
# Figure 1: Training vs Validation Loss (2x2 grid with all metrics)
# ============================================================================
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig)

# Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, history['train_accuracy'], 'b-', linewidth=2, label='Training Accuracy')
ax2.plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_title('Training vs Validation Accuracy', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# mIoU
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(epochs, history['train_mIoU'], 'b-', linewidth=2, label='Training mIoU')
ax3.plot(epochs, history['val_mIoU'], 'r-', linewidth=2, label='Validation mIoU')
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('mIoU (%)', fontsize=11)
ax3.set_title('Training vs Validation mIoU', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Learning curves (training only - zoomed)
ax4 = fig.add_subplot(gs[1, 1])
start_epoch = 20  # Skip first 20 epochs for better visualization
ax4.plot(epochs[start_epoch:], history['train_mIoU'][start_epoch:], 'b-', linewidth=2, label='Training mIoU')
ax4.plot(epochs[start_epoch:], history['val_mIoU'][start_epoch:], 'r-', linewidth=2, label='Validation mIoU')
ax4.set_xlabel('Epoch', fontsize=11)
ax4.set_ylabel('mIoU (%)', fontsize=11)
ax4.set_title(f'mIoU Training Curves (Epochs {start_epoch}-100, zoomed)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved training_curves.png")
plt.close()

# ============================================================================
# Figure 2: Per-Class IoU Comparison (Test Results)
# ============================================================================
class_names = ['Road', 'Sidewalk', 'Tree', 'Car', 'Fence', 'Pedestrian', 
               'Building', 'Pole', 'Sky', 'Bicycle', 'Sign', 'Void']

ious = eval_results['ious']
accuracies = eval_results['class_accuracies']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Per-class IoU
colors = ['#1f77b4' if iou > 30 else '#ff7f0e' if iou > 10 else '#d62728' for iou in ious]
bars1 = ax1.barh(class_names, ious, color=colors)
ax1.set_xlabel('IoU (%)', fontsize=11)
ax1.set_title('Per-Class IoU on Test Set', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 100)
for i, (bar, val) in enumerate(zip(bars1, ious)):
    ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
             va='center', fontsize=9)
ax1.grid(True, alpha=0.3, axis='x')

# Per-class Accuracy
colors = ['#1f77b4' if acc > 40 else '#ff7f0e' if acc > 15 else '#d62728' for acc in accuracies]
bars2 = ax2.barh(class_names, accuracies, color=colors)
ax2.set_xlabel('Accuracy (%)', fontsize=11)
ax2.set_title('Per-Class Accuracy on Test Set', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 100)
for i, (bar, val) in enumerate(zip(bars2, accuracies)):
    ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
             va='center', fontsize=9)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(plots_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved per_class_metrics.png")
plt.close()

# ============================================================================
# Figure 3: Test Set Results Summary
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Global\nAccuracy', 'Mean Class\nAccuracy', 'Mean IoU']
values = [
    eval_results['global_accuracy'],
    eval_results['mean_class_accuracy'],
    eval_results['mean_iou']
]
paper_targets = [89.4, 67.2, 65.3]  # From SegNet paper

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, values, width, label='Our Implementation', color='#1f77b4')
bars2 = ax.bar(x + width/2, paper_targets, width, label='Paper Baseline', color='#2ca02c')

ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Test Set Performance: Our Implementation vs Paper Baseline', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(plots_dir / 'test_results_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved test_results_comparison.png")
plt.close()

# ============================================================================
# Figure 4: Confusion Matrix and Class Imbalance Analysis
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Class distribution (estimated from class accuracies)
# High accuracy classes indicate they're well-represented
class_importance = np.array(accuracies) / np.array(accuracies).max() * 100

colors_dist = plt.cm.RdYlGn(class_importance / 100)
bars = ax1.barh(class_names, class_importance, color=colors_dist)
ax1.set_xlabel('Relative Representation Score (%)', fontsize=11)
ax1.set_title('Estimated Class Distribution in CamVid Test Set\n(Based on Per-Class Accuracy)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 105)
for i, (bar, val) in enumerate(zip(bars, class_importance)):
    ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', 
             va='center', fontsize=9)
ax1.grid(True, alpha=0.3, axis='x')

# IoU vs Accuracy scatter
ax2.scatter(accuracies, ious, s=200, alpha=0.6, c=range(len(class_names)), cmap='tab20')
for i, txt in enumerate(class_names):
    ax2.annotate(txt, (accuracies[i], ious[i]), fontsize=9, 
                xytext=(5, 5), textcoords='offset points')
ax2.set_xlabel('Class Accuracy (%)', fontsize=11)
ax2.set_ylabel('Class IoU (%)', fontsize=11)
ax2.set_title('Class-wise Accuracy vs IoU\n(Stricter IoU metric visible)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add diagonal reference line
min_val = min(min(accuracies), min(ious))
max_val = max(max(accuracies), max(ious))
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Accuracy=IoU')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(plots_dir / 'class_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved class_analysis.png")
plt.close()

print("\n" + "="*70)
print("PLOTS GENERATED SUCCESSFULLY")
print("="*70)
print(f"\nGenerated plots saved to: {plots_dir.absolute()}")
print("\nFiles created:")
print("  1. training_curves.png - Training vs validation metrics over epochs")
print("  2. per_class_metrics.png - Per-class IoU and accuracy on test set")
print("  3. test_results_comparison.png - Comparison with paper baseline")
print("  4. class_analysis.png - Class distribution and accuracy vs IoU analysis")
print("\n" + "="*70)

# Add value labels (only non-zero)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(plots_dir / 'test_results_comparison.pdf', dpi=300, bbox_inches='tight')
print("✓ Saved: test_results_comparison.pdf")
plt.close()

print("\n" + "="*60)
print("All plots generated successfully!")
print("="*60)

# Print summary statistics
print("\n### TRAINING STATISTICS ###")
print(f"Final Training Loss:        {history['train_loss'][-1]:.4f}")
print(f"Final Validation Loss:      {history['val_loss'][-1]:.4f}")
print(f"Final Training Accuracy:    {history['train_accuracy'][-1]:.2f}%")
print(f"Final Validation Accuracy:  {history['val_accuracy'][-1]:.2f}%")
print(f"Final Training mIoU:        {history['train_mIoU'][-1]:.2f}%")
print(f"Final Validation mIoU:      {history['val_mIoU'][-1]:.2f}%")

print("\n### TEST SET STATISTICS ###")
print(f"Global Accuracy:            {eval_results['global_accuracy']:.2f}%")
print(f"Mean IoU:                   {eval_results['mean_iou']:.2f}%")
print(f"Mean Class Accuracy:        {eval_results['mean_class_accuracy']:.2f}%")
print(f"Non-zero Class IoUs:        {sum(1 for x in eval_results['ious'] if x > 0.1)}/12")

print("\n### PAPER BASELINE COMPARISON ###")
print(f"Global Accuracy:   Test {eval_results['global_accuracy']:.1f}% vs Paper 89.4% ({eval_results['global_accuracy']-89.4:+.1f}%)")
print(f"Mean IoU:          Test {eval_results['mean_iou']:.1f}% vs Paper 65.3% ({eval_results['mean_iou']-65.3:+.1f}%)")
print(f"Class Avg Acc:     Test {eval_results['mean_class_accuracy']:.1f}% vs Paper 67.2% ({eval_results['mean_class_accuracy']-67.2:+.1f}%)")
