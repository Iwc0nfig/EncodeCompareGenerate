import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockUp(nn.Module):
    """
    Upsampling ResNet Block for the Decoder (The Hand).
    Uses Upsample + Conv instead of ConvTranspose to avoid checkerboard artifacts.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(ResidualBlockUp, self).__init__()
        self.scale_factor = scale_factor
        
        # Main Path
        # We do the upsampling explicitly in the forward pass
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut Path
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or scale_factor != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 1. Upsample (The "Expansion" step)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        
        # 2. Convolutions (The "Refining" step)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 3. Add Residual connection
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    


class MyDecoder(nn.Module):
    """
    Tool 2: The Concept Generator.
    Expands a Concept (z) into Reality (Image).
    """
    def __init__(self, emb_dim=128, out_channels=1):
        super(MyDecoder, self).__init__()
        
        # 1. Project Concept 'z' to a 7x7 spatial grid
        # We match the channel depth of the encoder's layer3 (128 channels)
        self.fc_start = nn.Linear(emb_dim, 128 * 7 * 7)
        self.unflatten = nn.Unflatten(1, (128, 7, 7))

        # 2. Upsampling Layers (Mirroring the Encoder)
        # 7x7 -> 14x14
        self.up1 = ResidualBlockUp(128, 64, scale_factor=2)
        
        # 14x14 -> 28x28
        self.up2 = ResidualBlockUp(64, 32, scale_factor=2)
        
        # 28x28 Refinement (No upsampling, just polishing)
        self.final_block = ResidualBlockUp(32, 32, scale_factor=1)
        
        # 3. Final Output (Map to pixels)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # Expand concept vector to feature map
        x = self.fc_start(z)
        x = self.unflatten(x)
        
        # Paint the image
        x = self.up1(x) # 14x14
        x = self.up2(x) # 28x28
        x = self.final_block(x)
        
        # Final pixels
        x = self.sigmoid(self.final_conv(x))
        return x