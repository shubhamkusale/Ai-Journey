import torch 
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
    ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))]
)

train_data = datasets.FashionMNIST(root='.data', train=True, download=True,transform=train_transform)
test_data = datasets.FashionMNIST(root='.data', train=False, download=True,transform=test_transform)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size = 64, shuffle =True)
val_loader = DataLoader(val_data , batch_size = 64, shuffle =False)
test_loader =DataLoader(test_data, batch_size = 64, shuffle =False)

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,padding=1) 
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,padding=1) 
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(32* 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)   

    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout3(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

model = ImprovedCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience= 3, factor=0.5)

best_val_loss = float('inf')
patience = 5
patience_counter = 0 
train_losses = []
val_losses = []

for epoch in range(50):
    model.train()
    total_train_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad(): 
        for images,labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct +=(predicted == labels).sum().item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = 100* correct/ total

    scheduler.step(avg_val_loss)

    print(f"Epoch :{epoch+1} | train_loss ={avg_train_loss:.4f}, val_loss{avg_val_loss}, val_acc{val_accuracy:.1f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(),'best_fashion_model.pth')
        patience_counter = 0
    else:
        patience_counter +=1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break 