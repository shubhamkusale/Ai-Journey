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

