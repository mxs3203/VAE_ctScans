import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary

from random import randint

from IPython.display import Image
from IPython.core.display import Image, display
import numpy as np
import torch.optim as optim
import matplotlib
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from VAE_CT.CT_DataLoader import CT_DataLoader
from VAE_CT.VAEModel import ConvVAE
from VAE_CT.utils import train, validate, save_loss_plot

matplotlib.style.use('ggplot')

transform = transforms.Compose([
    transforms.ToTensor(),
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set the learning parameters
lr = 0.001
epochs = 30
batch_size = 8
model = ConvVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


dataset = CT_DataLoader("../data/train/", transform)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=1, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=1, shuffle=True)

# a list to save all the reconstructed images in PyTorch grid format
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

for epoch in range(epochs):
    model.train()
    running_loss = []
    running_bce = []
    running_kld = []
    for images in trainLoader:

        images = images.to(device)
        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        loss, bce, kld = loss_fn(recon_images,
                                 images,
                                 mu,
                                 logvar)

        loss.backward()
        optimizer.step()
        running_loss.append(loss.cpu().detach())
        running_bce.append(bce.cpu().detach())
        running_kld.append(kld.cpu().detach())

    to_print = "Epoch[{}/{}] \n\tTraining Loss: {:.3f} {:.3f} {:.3f}".format(epoch + 1,
                                                                epochs, np.mean(running_loss) ,
                                                                np.mean(running_bce), np.mean(running_kld ))
    print(to_print)

    # Validation
    running_loss = []
    running_bce = []
    running_kld = []
    with torch.no_grad():
        model.eval()
        for images in valLoader:
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss, bce, kld = loss_fn(recon_images,
                                     images,
                                     mu,
                                     logvar)

            running_loss.append(loss.cpu().detach())
            running_bce.append(bce.cpu().detach())
            running_kld.append(kld.cpu().detach())
        to_print = "\tValidation Loss: {:.3f} {:.3f} {:.3f}".format(epoch + 1,
                                                                epochs, np.mean(running_loss) ,
                                                                np.mean(running_bce), np.mean(running_kld ))
        print(to_print)
print('TRAINING COMPLETE')