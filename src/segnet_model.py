"""
SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
Implementation based on: https://arxiv.org/abs/1511.00561
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EncoderBlock(nn.Module):
    """Single encoder block: Conv -> BatchNorm -> ReLU -> MaxPool"""
    
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x, pool_indices = self.pool(x)
        return x, pool_indices


class DecoderBlock(nn.Module):
    """Single decoder block: Unpool (using indices) -> Conv -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, pool_indices, output_size=None):
        # Unpool using the pooling indices from encoder
        x = self.unpool(x, pool_indices, output_size=output_size)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SegNetEncoder(nn.Module):
    """SegNet Encoder: VGG16-style encoder with 13 convolutional layers"""
    
    def __init__(self, in_channels=3, pretrained=True):
        super(SegNetEncoder, self).__init__()
        
        # Define encoder blocks (mimics VGG16 structure)
        self.block1_1 = EncoderBlock(in_channels, 64)
        self.block1_2 = EncoderBlock(64, 64)
        
        self.block2_1 = EncoderBlock(64, 128)
        self.block2_2 = EncoderBlock(128, 128)
        
        self.block3_1 = EncoderBlock(128, 256)
        self.block3_2 = EncoderBlock(256, 256)
        self.block3_3 = EncoderBlock(256, 256)
        
        self.block4_1 = EncoderBlock(256, 512)
        self.block4_2 = EncoderBlock(512, 512)
        self.block4_3 = EncoderBlock(512, 512)
        
        self.block5_1 = EncoderBlock(512, 512)
        self.block5_2 = EncoderBlock(512, 512)
        self.block5_3 = EncoderBlock(512, 512)
        
        # Initialize weights using He initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Store feature maps and pooling indices for decoder
        features = {}
        indices = {}
        
        # Block 1
        x, indices['pool1_1'] = self.block1_1(x)
        x, indices['pool1_2'] = self.block1_2(x)
        features['pool1'] = x
        
        # Block 2
        x, indices['pool2_1'] = self.block2_1(x)
        x, indices['pool2_2'] = self.block2_2(x)
        features['pool2'] = x
        
        # Block 3
        x, indices['pool3_1'] = self.block3_1(x)
        x, indices['pool3_2'] = self.block3_2(x)
        x, indices['pool3_3'] = self.block3_3(x)
        features['pool3'] = x
        
        # Block 4
        x, indices['pool4_1'] = self.block4_1(x)
        x, indices['pool4_2'] = self.block4_2(x)
        x, indices['pool4_3'] = self.block4_3(x)
        features['pool4'] = x
        
        # Block 5
        x, indices['pool5_1'] = self.block5_1(x)
        x, indices['pool5_2'] = self.block5_2(x)
        x, indices['pool5_3'] = self.block5_3(x)
        features['pool5'] = x
        
        return x, features, indices


class SegNetDecoder(nn.Module):
    """SegNet Decoder: Mirror structure of encoder with pooling indices-based upsampling"""
    
    def __init__(self, num_classes=11):
        super(SegNetDecoder, self).__init__()
        
        # Block 5 decoder
        self.block5_3 = DecoderBlock(512, 512)
        self.block5_2 = DecoderBlock(512, 512)
        self.block5_1 = DecoderBlock(512, 256)
        
        # Block 4 decoder
        self.block4_3 = DecoderBlock(256, 512)
        self.block4_2 = DecoderBlock(512, 512)
        self.block4_1 = DecoderBlock(512, 128)
        
        # Block 3 decoder
        self.block3_3 = DecoderBlock(128, 256)
        self.block3_2 = DecoderBlock(256, 256)
        self.block3_1 = DecoderBlock(256, 128)
        
        # Block 2 decoder
        self.block2_2 = DecoderBlock(128, 128)
        self.block2_1 = DecoderBlock(128, 64)
        
        # Block 1 decoder
        self.block1_2 = DecoderBlock(64, 64)
        self.block1_1 = DecoderBlock(64, num_classes)
        
        # Final classification layer
        self.classifier = nn.Conv2d(num_classes, num_classes, kernel_size=1)
    
    def forward(self, x, indices):
        # Block 5 decoder (reverse order)
        x = self.block5_3(x, indices['pool5_3'], output_size=None)
        x = self.block5_2(x, indices['pool5_2'], output_size=None)
        x = self.block5_1(x, indices['pool5_1'], output_size=None)
        
        # Block 4 decoder
        x = self.block4_3(x, indices['pool4_3'], output_size=None)
        x = self.block4_2(x, indices['pool4_2'], output_size=None)
        x = self.block4_1(x, indices['pool4_1'], output_size=None)
        
        # Block 3 decoder
        x = self.block3_3(x, indices['pool3_3'], output_size=None)
        x = self.block3_2(x, indices['pool3_2'], output_size=None)
        x = self.block3_1(x, indices['pool3_1'], output_size=None)
        
        # Block 2 decoder
        x = self.block2_2(x, indices['pool2_2'], output_size=None)
        x = self.block2_1(x, indices['pool2_1'], output_size=None)
        
        # Block 1 decoder
        x = self.block1_2(x, indices['pool1_2'], output_size=None)
        x = self.block1_1(x, indices['pool1_1'], output_size=None)
        
        # Final classification
        x = self.classifier(x)
        return x


class SegNet(nn.Module):
    """Complete SegNet architecture: Encoder + Decoder + Classifier"""
    
    def __init__(self, num_classes=11, in_channels=3, pretrained=True):
        super(SegNet, self).__init__()
        self.encoder = SegNetEncoder(in_channels=in_channels, pretrained=pretrained)
        self.decoder = SegNetDecoder(num_classes=num_classes)
        self.num_classes = num_classes
    
    def forward(self, x):
        input_shape = x.shape
        
        # Encoder
        x, features, indices = self.encoder(x)
        
        # Decoder
        x = self.decoder(x, indices)
        
        return x
    
    def forward_with_features(self, x):
        """Forward pass that also returns intermediate features (useful for analysis)"""
        # Encoder
        x, features, indices = self.encoder(x)
        
        # Decoder
        x = self.decoder(x, indices)
        
        return x, features


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = SegNet(num_classes=11, in_channels=3)
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Create dummy input
    x = torch.randn(1, 3, 360, 480)
    
    # Forward pass
    try:
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print("Model test successful!")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
