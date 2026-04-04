"""
SegNet Tutorial - Complete workflow demonstration
Run this to understand the full pipeline
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

# Import SegNet modules
from src.segnet_model import SegNet, count_parameters
from src.dataset import ToySegmentationDataset, create_dataloader
from src.utils import SegNetLoss, SegmentationMetrics


def tutorial_1_model_overview():
    """Tutorial 1: Understanding the SegNet Model"""
    print("\n" + "="*70)
    print("TUTORIAL 1: SegNet Model Overview")
    print("="*70)
    
    # Create model
    print("\n1. Creating SegNet model...")
    model = SegNet(num_classes=11, in_channels=3, pretrained=False)
    params = count_parameters(model)
    print(f"   ✅ Model created with {params:,} trainable parameters")
    
    # Model architecture
    print("\n2. Model Architecture:")
    print(f"   - Encoder: 13 convolutional layers (VGG16-based)")
    print(f"   - Decoder: 13 symmetric decoder layers")
    print(f"   - Pooling indices: Stored from encoder")
    print(f"   - Output: 11-class segmentation map")
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\n3. Model moved to device: {device}")
    
    return model, device


def tutorial_2_forward_pass(model, device):
    """Tutorial 2: Forward Pass and Output"""
    print("\n" + "="*70)
    print("TUTORIAL 2: Forward Pass")
    print("="*70)
    
    # Create dummy input
    batch_size, height, width = 2, 360, 480
    dummy_input = torch.randn(batch_size, 3, height, width).to(device)
    
    print(f"\n1. Input shape: {dummy_input.shape}")
    print(f"   (batch_size={batch_size}, channels=3, H={height}, W={width})")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\n2. Output shape: {output.shape}")
    print(f"   (batch_size={batch_size}, num_classes=11, H={height}, W={width})")
    
    # Get predictions
    predictions = torch.argmax(output, dim=1)
    print(f"\n3. Predictions shape: {predictions.shape}")
    print(f"   Values range: {predictions.min()} to {predictions.max()}")
    
    return output


def tutorial_3_dataset_loading():
    """Tutorial 3: Dataset and DataLoader"""
    print("\n" + "="*70)
    print("TUTORIAL 3: Dataset and DataLoader")
    print("="*70)
    
    # Create toy dataset
    print("\n1. Creating toy dataset...")
    dataset = ToySegmentationDataset(num_samples=20, num_classes=11)
    print(f"   ✅ Dataset created with {len(dataset)} samples")
    
    # Single sample
    print("\n2. Single sample:")
    image, mask = dataset[0]
    print(f"   Image shape: {image.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Mask unique classes: {torch.unique(mask).tolist()}")
    
    # DataLoader
    print("\n3. Creating DataLoader...")
    loader = create_dataloader(dataset, batch_size=4, shuffle=True)
    batch_images, batch_masks = next(iter(loader))
    
    print(f"   Batch image shape: {batch_images.shape}")
    print(f"   Batch mask shape: {batch_masks.shape}")
    print(f"   ✅ DataLoader ready for training")
    
    return loader


def tutorial_4_loss_and_metrics(model, device):
    """Tutorial 4: Loss Function and Metrics"""
    print("\n" + "="*70)
    print("TUTORIAL 4: Loss Function and Metrics")
    print("="*70)
    
    # Create loss function
    print("\n1. Creating loss function...")
    criterion = SegNetLoss(num_classes=11)
    print(f"   ✅ Cross-entropy loss with optional class balancing")
    
    # Create metrics
    print("\n2. Creating metrics...")
    metrics = SegmentationMetrics(num_classes=11)
    print(f"   ✅ Metrics available:")
    print(f"      - Global Accuracy")
    print(f"      - Class Average Accuracy")
    print(f"      - Mean IoU (Jaccard Index)")
    print(f"      - Per-class accuracies")
    
    # Dummy forward pass
    print("\n3. Computing loss on dummy batch...")
    dummy_input = torch.randn(2, 3, 360, 480).to(device)
    dummy_target = torch.randint(0, 11, (2, 360, 480)).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
    
    print(f"   Loss value: {loss.item():.4f}")
    
    # Compute metrics
    metrics.update(output, dummy_target)
    metrics_dict = metrics.get_metrics_dict()
    
    print(f"\n4. Metrics computed:")
    print(f"   Global Accuracy: {metrics_dict['global_accuracy']:.2f}%")
    print(f"   Class Avg Accuracy: {metrics_dict['mean_class_accuracy']:.2f}%")
    print(f"   Mean IoU: {metrics_dict['mean_iou']:.2f}%")


def tutorial_5_training_step(model, device):
    """Tutorial 5: Single Training Step"""
    print("\n" + "="*70)
    print("TUTORIAL 5: Single Training Step")
    print("="*70)
    
    # Optimizer
    print("\n1. Creating optimizer...")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print(f"   ✅ SGD optimizer with lr=0.01, momentum=0.9")
    
    # Loss function
    criterion = SegNetLoss(num_classes=11)
    
    # Dummy batch
    print("\n2. Creating dummy batch...")
    dummy_images = torch.randn(2, 3, 360, 480).to(device)
    dummy_labels = torch.randint(0, 11, (2, 360, 480)).to(device)
    print(f"   Images: {dummy_images.shape}")
    print(f"   Labels: {dummy_labels.shape}")
    
    # Training step
    print("\n3. Executing training step:")
    
    # Set model to training mode
    model.train()
    
    # Forward pass
    print("   a) Forward pass...")
    outputs = model(dummy_images)
    loss = criterion(outputs, dummy_labels)
    print(f"      Loss: {loss.item():.4f}")
    
    # Backward pass
    print("   b) Backward pass...")
    optimizer.zero_grad()
    loss.backward()
    
    # Optimizer step
    print("   c) Optimizer step...")
    optimizer.step()
    
    print(f"\n   ✅ Training step completed!")
    print(f"   Model weights have been updated")


def tutorial_6_full_training_loop():
    """Tutorial 6: Full Training Loop"""
    print("\n" + "="*70)
    print("TUTORIAL 6: Full Training Loop (Mini Example)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup
    print("\n1. Setting up training...")
    model = SegNet(num_classes=11).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = SegNetLoss(num_classes=11)
    
    # Dataset
    dataset = ToySegmentationDataset(num_samples=20, num_classes=11)
    loader = create_dataloader(dataset, batch_size=4, shuffle=True)
    print(f"   ✅ Model, optimizer, and dataloader ready")
    
    # Training
    print("\n2. Running 3 epochs...")
    num_epochs = 3
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        print(f"   Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
    
    print(f"\n   ✅ Training completed!")


def main():
    """Run all tutorials"""
    print("\n" + "🎓 "*35)
    print("SEGNET IMPLEMENTATION TUTORIAL")
    print("🎓 "*35)
    
    # Tutorial 1
    model, device = tutorial_1_model_overview()
    
    # Tutorial 2
    output = tutorial_2_forward_pass(model, device)
    
    # Tutorial 3
    loader = tutorial_3_dataset_loading()
    
    # Tutorial 4
    tutorial_4_loss_and_metrics(model, device)
    
    # Tutorial 5
    tutorial_5_training_step(model, device)
    
    # Tutorial 6
    tutorial_6_full_training_loop()
    
    print("\n" + "="*70)
    print("TUTORIAL COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python src/train.py --use_toy_dataset --epochs 10")
    print("2. Check training logs and results/")
    print("3. Evaluate: python src/evaluate.py --model_path models/custom_segnet/best_model.pth")
    print("4. For real dataset: Download CamVid and run without --use_toy_dataset")
    print("\n")


if __name__ == "__main__":
    main()
