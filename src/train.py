"""
Training script for SegNet
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segnet_model import SegNet, count_parameters
from src.dataset import ToySegmentationDataset, CamVidDataset, SegmentationTransform, create_dataloader
from src.utils import SegNetLoss, SegmentationMetrics


class SegNetTrainer:
    """Trainer class for SegNet"""
    
    def __init__(self, config):
        self.config = config
        # Use device specified in config, or auto-detect
        device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Initialize model
        self.model = SegNet(
            num_classes=config['num_classes'],
            in_channels=3,
            pretrained=False
        ).to(self.device)
        
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        # Loss and optimizer
        self.criterion = SegNetLoss(num_classes=config['num_classes'])
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Metrics
        self.train_metrics = SegmentationMetrics(num_classes=config['num_classes'])
        self.val_metrics = SegmentationMetrics(num_classes=config['num_classes'])
        
        # TensorBoard
        self.writer = SummaryWriter(config['log_dir'])
        
        # History
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'train_mIoU': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_mIoU': [],
        }
        
        self.best_val_miou = 0.0
        self.start_epoch = 0
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_miou = checkpoint.get('best_val_miou', 0.0)
            print(f"Loaded checkpoint from {checkpoint_path}")
            return True
        return False
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_miou': self.best_val_miou,
        }
        
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.train_metrics.update(outputs.detach(), labels)
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        epoch_metrics = self.train_metrics.get_metrics_dict()
        
        return avg_loss, epoch_metrics
    
    def validate(self, val_loader, epoch):
        """Validate on validation set"""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(progress_bar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                self.val_metrics.update(outputs, labels)
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(val_loader)
        epoch_metrics = self.val_metrics.get_metrics_dict()
        
        return avg_loss, epoch_metrics
    
    def train(self, train_loader, val_loader=None):
        """Complete training loop"""
        num_epochs = self.config['num_epochs']
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("="*60)
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Logging
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_metrics['global_accuracy']:.2f}%")
            print(f"Train mIoU: {train_metrics['mean_iou']:.2f}%")
            
            # Validation
            if val_loader is not None:
                val_loss, val_metrics = self.validate(val_loader, epoch)
                
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Accuracy: {val_metrics['global_accuracy']:.2f}%")
                print(f"Val mIoU: {val_metrics['mean_iou']:.2f}%")
                
                # Update learning rate
                self.scheduler.step(val_metrics['mean_iou'])
                
                # Save best model
                if val_metrics['mean_iou'] > self.best_val_miou:
                    self.best_val_miou = val_metrics['mean_iou']
                    self.save_checkpoint(epoch, is_best=True)
                
                # Log to TensorBoard
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_metrics['global_accuracy'], epoch)
                self.writer.add_scalar('Accuracy/val', val_metrics['global_accuracy'], epoch)
                self.writer.add_scalar('mIoU/train', train_metrics['mean_iou'], epoch)
                self.writer.add_scalar('mIoU/val', val_metrics['mean_iou'], epoch)
                
                # History
                self.history['train_loss'].append(train_loss)
                self.history['train_accuracy'].append(train_metrics['global_accuracy'])
                self.history['train_mIoU'].append(train_metrics['mean_iou'])
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_metrics['global_accuracy'])
                self.history['val_mIoU'].append(val_metrics['mean_iou'])
            else:
                self.save_checkpoint(epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch)
        
        self.writer.close()
        print(f"\nTraining completed!")
        return self.history
    
    def save_history(self, save_path):
        """Save training history to JSON"""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Saved training history to {save_path}")


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SegNet')
    parser.add_argument('--model', type=str, default='custom', choices=['custom', 'official'],
                       help='Model type')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--dataset_path', type=str, default='SegNet-Tutorial/CamVid/',
                       help='Path to dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='models/custom_segnet/',
                       help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs/segnet_training/',
                       help='Log directory')
    parser.add_argument('--results_dir', type=str, default='results/',
                       help='Results directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--num_classes', type=int, default=11,
                       help='Number of classes')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint frequency')
    parser.add_argument('--use_toy_dataset', action='store_true',
                       help='Use toy dataset for testing')
    
    args = parser.parse_args()
    
    # Config
    config = {
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'save_freq': args.save_freq,
        'device': args.device,
    }
    
    # Initialize trainer
    trainer = SegNetTrainer(config)
    
    # Create datasets
    if args.use_toy_dataset:
        print("Using toy dataset for quick testing...")
        train_dataset = ToySegmentationDataset(num_samples=50, num_classes=args.num_classes, split='train')
        val_dataset = ToySegmentationDataset(num_samples=20, num_classes=args.num_classes, split='val')
    else:
        print(f"Using dataset from {args.dataset_path}...")
        transform = SegmentationTransform(augment=True)
        train_dataset = CamVidDataset(args.dataset_path, split='train', transform=transform)
        val_dataset = CamVidDataset(args.dataset_path, split='val', transform=transform)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = create_dataloader(train_dataset, batch_size=config['batch_size'], 
                                     shuffle=True, num_workers=0)
    val_loader = create_dataloader(val_dataset, batch_size=config['batch_size'], 
                                   shuffle=False, num_workers=0)
    
    # Train
    print(f"\nStarting training on {args.device}...")
    start_time = time.time()
    history = trainer.train(train_loader, val_loader)
    training_time = time.time() - start_time
    
    print(f"\nTotal training time: {training_time/3600:.2f} hours")
    
    # Save history
    os.makedirs(args.results_dir, exist_ok=True)
    trainer.save_history(os.path.join(args.results_dir, 'training_history.json'))


if __name__ == "__main__":
    main()
