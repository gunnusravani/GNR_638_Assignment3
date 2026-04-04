"""
SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
Implementation based on: https://arxiv.org/abs/1511.00561
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    """Convolutional block: Conv -> BatchNorm -> ReLU (no pooling)"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    """Encoder block: Multiple convolutions followed by a single MaxPool with indices"""
    
    def __init__(self, in_channels, out_channels, num_convs=2):
        super(EncoderBlock, self).__init__()
        self.convs = nn.ModuleList()
        
        # First conv changes channels
        self.convs.append(ConvBlock(in_channels, out_channels))
        # Remaining convs keep same channel
        for _ in range(num_convs - 1):
            self.convs.append(ConvBlock(out_channels, out_channels))
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x, pool_indices = self.pool(x)
        return x, pool_indices


class SegNetEncoder(nn.Module):
    """SegNet Encoder: VGG16-style encoder with 13 convolutional layers and 4 pooling ops"""
    
    def __init__(self, in_channels=3, pretrained=True):
        super(SegNetEncoder, self).__init__()
        
        # Define encoder blocks with correct pooling (1 pooling per block)
        # Each block contains multiple conv layers followed by ONE pooling
        self.block1 = EncoderBlock(in_channels, 64, num_convs=2)   # 2 conv layers
        self.block2 = EncoderBlock(64, 128, num_convs=2)           # 2 conv layers
        self.block3 = EncoderBlock(128, 256, num_convs=3)          # 3 conv layers
        self.block4 = EncoderBlock(256, 512, num_convs=3)          # 3 conv layers
        self.block5 = EncoderBlock(512, 512, num_convs=3)          # 3 conv layers
        
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
        # Store feature maps, pooling indices, and shapes for decoder
        features = {}
        indices = []
        sizes = []
        
        # Block 1: 360x480 -> 180x240
        sizes.append(x.size())
        x, pool_idx = self.block1(x)
        indices.append(pool_idx)
        features['pool1'] = x
        
        # Block 2: 180x240 -> 90x120
        sizes.append(x.size())
        x, pool_idx = self.block2(x)
        indices.append(pool_idx)
        features['pool2'] = x
        
        # Block 3: 90x120 -> 45x60
        sizes.append(x.size())
        x, pool_idx = self.block3(x)
        indices.append(pool_idx)
        features['pool3'] = x
        
        # Block 4: 45x60 -> 22x30
        sizes.append(x.size())
        x, pool_idx = self.block4(x)
        indices.append(pool_idx)
        features['pool4'] = x
        
        # Block 5: 22x30 -> 11x15
        sizes.append(x.size())
        x, pool_idx = self.block5(x)
        indices.append(pool_idx)
        features['pool5'] = x
        
        return x, features, indices, sizes


class SegNetDecoder(nn.Module):
    """SegNet Decoder: Mirror structure of encoder with pooling indices-based upsampling"""
    
    def __init__(self, num_classes=11):
        super(SegNetDecoder, self).__init__()
        
        # Block 5 decoder (mirror of block 5 encoder: 3 convs + unpool)
        self.block5_1 = ConvBlock(512, 512)
        self.block5_2 = ConvBlock(512, 512)
        self.block5_3 = ConvBlock(512, 512)
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        # Block 4 decoder (mirror of block 4 encoder: 3 convs + unpool)
        self.block4_1 = ConvBlock(512, 512)
        self.block4_2 = ConvBlock(512, 512)
        self.block4_3 = ConvBlock(512, 256)
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        # Block 3 decoder (mirror of block 3 encoder: 3 convs + unpool)
        self.block3_1 = ConvBlock(256, 256)
        self.block3_2 = ConvBlock(256, 256)
        self.block3_3 = ConvBlock(256, 128)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        # Block 2 decoder (mirror of block 2 encoder: 2 convs + unpool)
        self.block2_1 = ConvBlock(128, 128)
        self.block2_2 = ConvBlock(128, 64)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        # Block 1 decoder (mirror of block 1 encoder: 2 convs + unpool)
        self.block1_1 = ConvBlock(64, 64)
        self.block1_2 = ConvBlock(64, num_classes)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
    
    def forward(self, x, indices, sizes):
        # indices is a list: [pool1_idx, pool2_idx, pool3_idx, pool4_idx, pool5_idx]
        # sizes is a list of feature map sizes before each pooling
        # We process in reverse order
        
        # Block 5 decoder (unpool 5, then 3 convs)
        x = self.unpool5(x, indices[4], output_size=sizes[4])  # back to pool4 size
        x = self.block5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        
        # Block 4 decoder (unpool 4, then 3 convs)
        x = self.unpool4(x, indices[3], output_size=sizes[3])  # back to pool3 size
        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block4_3(x)
        
        # Block 3 decoder (unpool 3, then 3 convs)
        x = self.unpool3(x, indices[2], output_size=sizes[2])  # back to pool2 size
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        
        # Block 2 decoder (unpool 2, then 2 convs)
        x = self.unpool2(x, indices[1], output_size=sizes[1])  # back to pool1 size
        x = self.block2_1(x)
        x = self.block2_2(x)
        
        # Block 1 decoder (unpool 1, then 2 convs)
        x = self.unpool1(x, indices[0], output_size=sizes[0])  # back to input size
        x = self.block1_1(x)
        x = self.block1_2(x)
        
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
        x, features, indices, sizes = self.encoder(x)
        
        # Decoder
        x = self.decoder(x, indices, sizes)
        
        return x
    
    def forward_with_features(self, x):
        """Forward pass that also returns intermediate features (useful for analysis)"""
        # Encoder
        x, features, indices, sizes = self.encoder(x)
        
        # Decoder
        x = self.decoder(x, indices, sizes)
        
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
