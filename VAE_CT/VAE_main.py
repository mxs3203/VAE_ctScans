import random

import cv2 as cv
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from VAE_CT.CT_DataLoader import CT_DataLoader
from VAE_CT.VAEModel import ConvVAE

matplotlib.style.use('ggplot')

transform = transforms.Compose([
    transforms.ToTensor()
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set the learning parameters
lr = 0.001
epochs = 500
batch_size = 16
model = ConvVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCELoss(reduction='sum')

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

def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, KLD

def train():
    model.train()
    running_loss = []
    running_bce = []
    running_kld = []
    for images in trainLoader:
        images = images.to(device)
        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)

        loss = criterion(recon_images, images)
        loss,kld = final_loss(loss,mu, logvar)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.cpu().detach())
        running_kld.append(kld.cpu().detach())

    to_print = "Epoch[{}/{}] \n\tTraining Loss: {:.3f} KLD: {:.3f}".format(epoch + 1,
                                                               epochs,
                                                               np.mean(running_loss),
                                                               np.mean(running_kld))
    print(to_print)

def validate():
    # Validation
    running_loss = []
    running_kld = []
    rand = random.randint(0, batch_size)
    with torch.no_grad():
        model.eval()
        for images in valLoader:
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss = criterion(recon_images, images)
            loss,kld = final_loss(loss, mu, logvar)

            running_loss.append(loss.cpu().detach())
            running_kld.append(kld.cpu().detach())
        to_print = "\tValidation Loss: {:.3f} KLD: {:.3f}".format(
                                                       np.mean(running_loss),
                                                       np.mean(running_kld))
        print(to_print)
    return recon_images[rand, :, :, :]

for epoch in range(epochs):
    train()
    random_img = validate()
    for i in range(60):
        img = cv.normalize(random_img[:, :, i], None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
        plt.imshow(img, cmap=plt.cm.bone)
        plt.show()


print('TRAINING COMPLETE')