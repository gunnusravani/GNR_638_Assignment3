"""
Quick test script to verify SegNet implementation
"""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    try:
        from src.segnet_model import SegNet, count_parameters
        from src.dataset import ToySegmentationDataset, create_dataloader
        from src.utils import SegmentationMetrics, SegNetLoss
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_model_creation():
    """Test SegNet model creation"""
    print("\nTesting model creation...")
    try:
        from src.segnet_model import SegNet, count_parameters
        
        model = SegNet(num_classes=11, in_channels=3, pretrained=False)
        params = count_parameters(model)
        
        print(f"✅ Model created successfully")
        print(f"   Total parameters: {params:,}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"   Model moved to device: {device}")
        
        return True, model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model):
    """Test forward pass through model"""
    print("\nTesting forward pass...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, 360, 480).to(device)
        
        print(f"   Input shape: {x.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"   Output shape: {output.shape}")
        print(f"✅ Forward pass successful")
        
        # Check output
        assert output.shape == (batch_size, 11, 360, 480), "Output shape mismatch!"
        assert not torch.isnan(output).any(), "NaN in output!"
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset loading"""
    print("\nTesting dataset...")
    try:
        from src.dataset import ToySegmentationDataset, create_dataloader
        
        # Create toy dataset
        dataset = ToySegmentationDataset(num_samples=10, num_classes=11)
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Get single sample
        img, mask = dataset[0]
        print(f"   Image shape: {img.shape}")
        print(f"   Mask shape: {mask.shape}")
        
        # Create dataloader
        loader = create_dataloader(dataset, batch_size=2, shuffle=True)
        batch_img, batch_mask = next(iter(loader))
        print(f"   Batch image shape: {batch_img.shape}")
        print(f"   Batch mask shape: {batch_mask.shape}")
        
        print(f"✅ Dataset and DataLoader working")
        
        return True
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation"""
    print("\nTesting metrics...")
    try:
        from src.utils import SegmentationMetrics
        
        metrics = SegmentationMetrics(num_classes=11)
        
        # Create dummy predictions and targets
        batch_size, height, width = 2, 360, 480
        logits = torch.randn(batch_size, 11, height, width)
        targets = torch.randint(0, 11, (batch_size, height, width))
        
        # Update metrics
        metrics.update(logits, targets)
        
        # Get metrics
        metrics_dict = metrics.get_metrics_dict()
        print(f"✅ Metrics computed:")
        print(f"   Global Accuracy: {metrics_dict['global_accuracy']:.2f}%")
        print(f"   Mean Class Accuracy: {metrics_dict['mean_class_accuracy']:.2f}%")
        print(f"   Mean IoU: {metrics_dict['mean_iou']:.2f}%")
        
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss():
    """Test loss computation"""
    print("\nTesting loss function...")
    try:
        from src.utils import SegNetLoss
        
        criterion = SegNetLoss(num_classes=11)
        
        # Create dummy data
        batch_size, height, width = 2, 360, 480
        outputs = torch.randn(batch_size, 11, height, width)
        targets = torch.randint(0, 11, (batch_size, height, width))
        
        # Compute loss
        loss = criterion(outputs, targets)
        print(f"✅ Loss computed: {loss.item():.4f}")
        
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss is NaN"
        
        return True
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    try:
        import torch.optim as optim
        from src.segnet_model import SegNet
        from src.utils import SegNetLoss
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model and optimizer
        model = SegNet(num_classes=11).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = SegNetLoss(num_classes=11)
        
        # Create dummy data
        batch_size = 2
        images = torch.randn(batch_size, 3, 360, 480).to(device)
        labels = torch.randint(0, 11, (batch_size, 360, 480)).to(device)
        
        # Forward pass
        model.train()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✅ Training step successful")
        print(f"   Loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("SEGNET IMPLEMENTATION TEST SUITE")
    print("="*60)
    
    print(f"\nPython version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['model_creation'], model = test_model_creation()
    
    if model is not None:
        results['forward_pass'] = test_forward_pass(model)
    
    results['dataset'] = test_dataset()
    results['metrics'] = test_metrics()
    results['loss'] = test_loss()
    results['training_step'] = test_training_step()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! SegNet is ready for training.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
