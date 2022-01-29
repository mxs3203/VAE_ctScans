import torch.nn as nn
import torch

class AE(nn.Module):
    def __init__(self, inputsize):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(inputsize, 512),nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, inputsize),nn.ReLU()
        )


    def forward(self, x):

        x = self.decoder(x)
        reconstructed = self.encoder(x)

        return reconstructed