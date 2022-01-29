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
    def __init__(self, image_channels=60, h_dim=952576, z_dim=1):
        super(ConvVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.z_mean = nn.Linear(h_dim, z_dim)
        self.z_log_var = nn.Linear(h_dim, z_dim)


        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            UnFlatten(8, 256, 300, 300),
            nn.ConvTranspose2d(256, 128, kernel_size=4,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2 ),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        eps = torch.randn(mu.size(0), mu.size(1)).to(mu.get_device())
        z = mu + eps * torch.exp(logvar/2.)
        return z

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decoder(z), z_mean, z_log_var
