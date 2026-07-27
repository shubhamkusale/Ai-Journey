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

class BatchNormCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,padding=1) 
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.relu()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.relu2 = nn.relu()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(32* 7 * 7, 128)
        self.relu3 = nn.relu()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)   

    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout3(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

