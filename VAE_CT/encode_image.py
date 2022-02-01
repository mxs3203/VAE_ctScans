import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from VAE_CT.CT_DataLoader import CT_DataLoader
from VAE_CT.VAEModel import ConvVAE
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = CT_DataLoader("../data/train/", transform)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainLoader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)


model = ConvVAE()
model.load_state_dict(torch.load("../saved_models/vae_model_ep_199.pt"))
model.eval()


for images in trainLoader:
    images = images.to(device)
    recon_images, mu, logvar = model(images)
    for i in range(60):
        im = np.array(recon_images.squeeze(dim=0).cpu().detach())
        plt.imshow(im[i, :, :], cmap='gray')
        plt.show()
    break
