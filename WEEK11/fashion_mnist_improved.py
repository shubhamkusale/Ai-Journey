import torch 
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import dataloader
import matplotlib.pyplot as plt

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
    ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.normalize((0.5),(0.5))]
)

train_data = datasets.FashionMNIST(root='.data', train=True, download=True,transform=train_transform)
test_data = datasets.FashionMNIST(root='.data', train=False, download=True,transform=train_transform)

train_Size = int(0.8 * len(train_data))
val_size = len(train_data) - train_Size
train_data, val_data = torch.utils.data.random_split(train_data[train_Size,val_size])

train_loader = dataloader(train_data, batch_size = 64, shuffle =True)
val_loader = dataloader(val_data , batch_size = 64, shuffle =False)
test_loader =dataloader(test_data, batch_size = 64, shuffle =False)