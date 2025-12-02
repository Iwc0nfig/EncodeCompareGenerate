import torch.nn as nn
from my_nn.my_encoder import MyEncoder
from my_nn.my_decoder import MyDecoder

        

class RichNet(nn.Module):
    def __init__(self, in_channels=1, emb_dim=128):
        super(RichNet, self).__init__()
        # 1. The Eye (Encoder) - Reusing your existing architecture
        self.encoder = MyEncoder(in_channels, emb_dim)

        # 2. The Sketchpad (Decoder) - Adjusted for 28x28 Output
        self.decoder = MyDecoder(emb_dim, in_channels)
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon