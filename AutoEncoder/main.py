import pickle

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2 as cv
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

from AutoEncoder.model import AE
from VAE_CT.CT_DataLoader import CT_DataLoader

matplotlib.style.use('ggplot')

transform = transforms.Compose([
    transforms.ToTensor(),
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set the learning parameters
lr = 0.001
epochs = 30
batch_size = 32
model = AE(50*50*60).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


dataset = CT_DataLoader("../data/train_small/", transform)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

criterion = nn.MSELoss()

train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=1, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size, num_workers=1, shuffle=False)



for epoch in range(epochs):
    model.train()
    running_loss = []

    img = None
    for images in trainLoader:
        images = images.view(-1, 50*50*60).to(device)
        images = images.to(device)
        optimizer.zero_grad()
        recon_images = model(images)
        img = recon_images.cpu().detach()
        loss = criterion(recon_images, images)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.cpu().detach())

    to_print = "Epoch[{}/{}] \n\tTraining Loss: {:.3f} ".format(epoch + 1,epochs, np.mean(running_loss) )
    print(to_print)

    # Validation
    running_loss = []
    with torch.no_grad():
        model.eval()
        for images in valLoader:
            images = images.view(-1, 50*50*60).to(device)
            images = images.to(device)
            recon_images = model(images)
            loss = criterion(recon_images, images)

            running_loss.append(loss.cpu().detach())
        to_print = "\tValidation Loss: {:.3f} ".format(epoch + 1,epochs, np.mean(running_loss) )
        print(to_print)
        img = np.reshape(img, (batch_size, 50, 50, 60))
        img = torch.squeeze(img, dim=1)
        #img = cv.normalize(img[:, :, 30], None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        plt.imshow(img[0, :, :, 30], cmap=plt.cm.bone)
        plt.show()
print('TRAINING COMPLETE')