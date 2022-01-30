import torch.nn as nn
import torch

class AE(nn.Module):
    def __init__(self, image_channels=60):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=4), # 512/4=128, (B, 64, 128, 128)
            nn.ReLU(),
            nn.Conv2d(64, 80, kernel_size=4,stride=4), # 128/4=32 (B, 80, 32, 32)
            nn.ReLU(),
            nn.Conv2d(80, 100, kernel_size=4,stride=4), #32/4=8 (B, 100, 8, 8)
            nn.ReLU(),
            nn.Conv2d(100, 120, kernel_size=8,stride=8), #8/8=1 (B, 120, 1, 1)
            nn.ReLU(),
        )


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(120, 100, kernel_size=8,stride=8),
            nn.ReLU(),
            nn.ConvTranspose2d(100, 80, kernel_size=4,stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(80, 64, kernel_size=4,stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=4 ),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x