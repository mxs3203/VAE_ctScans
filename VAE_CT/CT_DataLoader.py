import numpy as np
import pickle
from torch.utils.data import Dataset
import torch
import glob
from skimage.transform import resize

class CT_DataLoader(Dataset):

    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.data = []  # Creates an empty list
        for f in glob.glob("{}/*.p".format(folder)):
            with (open(f, "rb")) as openfile:
                try:
                    arr = np.array(pickle.load(openfile))
                    self.data.append(arr)
                except EOFError:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = np.array(self.data[idx],dtype="uint8")

        if self.transform:
            x = self.transform(x)

        return x