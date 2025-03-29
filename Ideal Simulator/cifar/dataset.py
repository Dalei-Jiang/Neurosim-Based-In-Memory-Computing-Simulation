# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:34:04 2024

@author: dalei
"""

# import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import scipy
import numpy as np
import torch
# from skimage.transform import resize
import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

X_file = 'X_GaNmul3_VSHS'
y_file = 'y_GaN_hex'

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=int)

def loading(datatype, batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    Xname = X_file
    yname = y_file
    ds = []
    images = np.load(f"./resource/{Xname}.npy")
    labels = np.load(f"./resource/{yname}.npy")
    sample_number = images.shape[0]
    ratio1 = 0.8
    ratio2 = 1.0
    images_train = images[:int(sample_number*ratio1)]
    images_test = images[int(sample_number*ratio1):int(sample_number*ratio2)]
    labels_train = labels[:int(sample_number*ratio1)]
    labels_test = labels[int(sample_number*ratio1):int(sample_number*ratio2)]
    dataset_train = CustomDataset(images_train, labels_train)
    dataset_test = CustomDataset(images_test, labels_test)
    if train:
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)
        ds.append(train_loader)
    if val:
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    
    return ds






