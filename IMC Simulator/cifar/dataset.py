# import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import scipy
import numpy as np
import torch
from skimage.transform import resize
# Dalei Jiang
# This dataset imports cifar10-data and cifar100-data
# def replace_with_index(arr):
#     # Find the unique elements and their sorted order
#     unique_elements = np.unique(arr)
#     # Create a dictionary mapping each unique element to its index
#     element_to_index = {element: idx for idx, element in enumerate(unique_elements)}
#     # Replace each element in the array with its corresponding index
#     replaced_array = np.vectorize(element_to_index.get)(arr)
    
#     # Print the dictionary
#     print(element_to_index)
    
#     return replaced_array


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

def loading(datatype, batch_size, label, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    
    # CIFAR-10 dataset
    if datatype == 'cifar10':
        data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
        num_workers = kwargs.setdefault('num_workers', 1)
        kwargs.pop('input_size', None)
        print("Building CIFAR-10 data loader with {} workers".format(num_workers))
        ds = []
        if train:
            train_loader = DataLoader(
                datasets.CIFAR10(
                    root=data_root, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Pad(4),  # Padding 4 zeros edge
                        transforms.RandomCrop(32),  # Cut 32*32 patches randomly
                        transforms.RandomHorizontalFlip(), # Horizontal fipping with 50% possibility
                        transforms.ToTensor(),      # Switching to the Tensor
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalized the data to mean 0.5 and variance 0.5
                    ])),
                batch_size=batch_size, shuffle=True, **kwargs)
    
            ds.append(train_loader)
            
        if val:
            test_loader = DataLoader(
                datasets.CIFAR10(
                    root=data_root, train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
                batch_size=batch_size, shuffle=False, **kwargs)
            ds.append(test_loader)
        ds = ds[0] if len(ds) == 1 else ds
        
    
    # MNIST dataset    
    elif datatype == 'MNIST':
        data_root = os.path.expanduser(os.path.join(data_root, 'MNIST-data'))
        num_workers = kwargs.setdefault('num_workers', 1)
        kwargs.pop('input_size', None)
        print("Building MNIST data loader with {} workers".format(num_workers))
        ds = []
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if train:
            train_loader = DataLoader(
                datasets.MNIST(
                    root=data_root, train=True, download=True,
                    transform=transform),
                batch_size=batch_size, shuffle=True, **kwargs)
            ds.append(train_loader)
        if val:
            test_loader = DataLoader(
                datasets.MNIST(
                    root=data_root, train=False, download=True,
                    transform=transform),
                batch_size=batch_size, shuffle=False, **kwargs)
            ds.append(test_loader)
        ds = ds[0] if len(ds) == 1 else ds
    
    
    # CIFAR100 dataset    
    elif datatype == 'cifar100':
        data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
        num_workers = kwargs.setdefault('num_workers', 1)
        kwargs.pop('input_size', None)
        print("Building ImageNet data loader with {} workers".format(num_workers))
        ds = []
        
        transform_train = transforms.Compose([
            transforms.Pad(4),  # Padding 4 zeros edge
            transforms.RandomCrop(32),  # Cut 32*32 patches randomly
            transforms.RandomHorizontalFlip(), # Horizontal fipping with 50% possibility
            transforms.ToTensor(),      # Switching to the Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalized the data to mean 0.5 and variance 0.5
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),      # Switching to the Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalized the data to mean 0.5 and variance 0.5
        ])
        
        if train:
            train_loader = DataLoader(
                datasets.CIFAR100(
                    root=data_root, train=True, download=True,
                    transform=transform_train),
                batch_size=batch_size, shuffle=True, **kwargs)
            # subset_indices = torch.arange(0, 100000)  # 选择前 100,000 张图片
            # subset_data = torch.utils.data.Subset(imagenet_data, subset_indices)    
            
            ds.append(train_loader)
        if val:
            test_loader = DataLoader(
                datasets.CIFAR100(
                    root=data_root, train=False, download=True,
                    transform=transform_test),
                batch_size=batch_size, shuffle=False, **kwargs)
            ds.append(test_loader)
        ds = ds[0] if len(ds) == 1 else ds

    else:
        if datatype == 'GaN':
            Xname = f'X_GaNmul3_{label}'
            yname = 'y_GaN_hex'
        elif datatype == 'ScAlN':
            Xname = f'X_ScAlNmul3_{label}'
            yname = 'y_ScAlN_new'
        elif datatype == 'ScAlN_GaN':
            Xname = 'X_ScAlN_GaN'
            yname = 'y_ScAlN_GaN'
        ds = []
        images = np.load(f"../../resource/{Xname}.npy")
        print(f"{Xname} is loaded!")
        labels = np.load(f"../../resource/{yname}.npy")
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






