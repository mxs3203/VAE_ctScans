import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class UnFlatten(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)



class ConvVAE(nn.Module):
    def __init__(self, image_channels=60, h_dim=120, z_dim=120):
        super(ConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=4), # 512/4=128, (B, 64, 128, 128)
            nn.ReLU(),
            nn.Conv2d(64, 80, kernel_size=4,stride=4), # 128/4=32 (B, 80, 32, 32)
            nn.ReLU(),
            nn.Conv2d(80, 100, kernel_size=4,stride=4), #32/4=8 (B, 100, 8, 8)
            nn.ReLU(),
            nn.Conv2d(100, 120, kernel_size=8,stride=8), #8/8=1 (B, 120, 1, 1)
            nn.ReLU(),
            nn.Flatten()
        )

        self.z_mean = nn.Linear(h_dim, z_dim)
        self.z_log_var = nn.Linear(h_dim, z_dim)


        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.ReLU(),
            UnFlatten(-1, 120, 1, 1),
            nn.ConvTranspose2d(120, 100, kernel_size=8,stride=8),
            nn.ReLU(),
            nn.ConvTranspose2d(100, 80, kernel_size=4,stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(80, 64, kernel_size=4,stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=4 ),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        eps = torch.randn(mu.size(0), mu.size(1)).to(mu.get_device())
        z = mu + eps * torch.exp(logvar/0.5)
        return z

    def encode_img(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z = self.reparameterize(z_mean, z_log_var)
        return x,z, z_mean, z_log_var

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decoder(z), z_mean, z_log_var
