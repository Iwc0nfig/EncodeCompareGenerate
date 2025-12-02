import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # NOTE: Using InstanceNorm2d instead of BatchNorm2d
        # affine=True allows it to learn a scaling factor, keeping some flexibility
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MyEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, emb_dim: int = 128):
        super(MyEncoder, self).__init__()
        
        # --- ENCODER ---
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # ResNet Backbone
        self.layer1 = self._make_layer(32, 32, stride=1)  # 28x28
        self.layer2 = self._make_layer(32, 64, stride=2)  # 14x14
        self.layer3 = self._make_layer(64, 128, stride=2) # 7x7
        self.layer4 = self._make_layer(128, 256, stride=2) # 4x4


        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- EMBEDDING HEAD ---
        # This maps the features to the final vector 'z'
        self.fc_enc = nn.Linear(256, emb_dim)
        self.bn_enc = nn.BatchNorm1d(emb_dim)
        

    def _make_layer(self, in_channels, out_channels, stride):
        return ResidualBlock(in_channels, out_channels, stride)

    

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.pool(x).flatten(1)
        
        x = self.fc_enc(x)
        x = self.bn_enc(x)
        
        # CRITICAL: L2 Normalize
        # This projects 'z' onto the sphere. 
        # This makes it compatible with Cosine Similarity in memory.py
        return F.normalize(x, p=2, dim=1)
        