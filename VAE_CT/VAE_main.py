import random

import cv2 as cv
import pandas
import torch
import numpy as np
import torch.optim as optim
import matplotlib
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from VAE_CT.CT_DataLoader import CT_DataLoader
from VAE_CT.VAEModel import ConvVAE

matplotlib.style.use('ggplot')

tsne = TSNE(n_components=2, verbose=0, random_state=123)

transform = transforms.Compose([
    transforms.ToTensor()
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set the learning parameters
lr = 0.001
epochs = 200
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

def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, KLD

def train():
    model.train()
    running_loss = []
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
    return np.mean(running_loss)

def validate():
    # Validation
    running_loss = []
    running_kld = []
    #rand = random.randint(0, batch_size-1)
    with torch.no_grad():
        model.eval()
        for images in valLoader:
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss = criterion(recon_images, images)
            loss, kld = final_loss(loss, mu, logvar)

            running_loss.append(loss.cpu().detach())
            running_kld.append(kld.cpu().detach())
        to_print = "\tValidation Loss: {:.3f} KLD: {:.3f}".format(
                                                       np.mean(running_loss),
                                                       np.mean(running_kld))
        print(to_print)
    return recon_images[0, :, :, :]

def plot_latent():

    total = pandas.DataFrame()
    for x in valLoader:
        x, z, z_mean, z_log_var = model.encode_img(x.to(device))
        z = z.to('cpu').detach().numpy()
        z = pandas.DataFrame(z)
        total = pandas.concat([total ,z], ignore_index=True)

    tsne_results = tsne.fit_transform(total)
    tsne_results = pandas.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'])
    plt.show()


def visualize_recon(randomimg):
    num_row = 10
    num_col = 6  # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(2 * num_col, 2 * num_row))
    for i in range(60):
        ax = axes[i // num_col, i % num_col]
        im = np.array(randomimg[i, :, :])
        ax.imshow(im, cmap='gray')
    plt.tight_layout()
    plt.show()

loss = []
for epoch in range(epochs):
    train_loss = train()
    loss.append(train_loss)
    recon_image = validate()
    visualize_recon(recon_image.cpu().detach())
    plot_latent()
    torch.save(model.state_dict(), "../saved_models/vae_model_ep_{}.pt".format(epoch))


plt.plot(range(epochs), loss)
plt.title("Total Loss")
plt.show()


print('TRAINING COMPLETE')