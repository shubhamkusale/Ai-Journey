import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # detects edges
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128) 
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))  
        x = self.pool2(self.relu2(self.conv2(x)))  
        x = x.view(-1, 32 * 7 * 7)               
        x = self.relu3(self.fc1(x))               
        x = self.fc2(x)                           
        return x

# 3. SETUP
model = CNN()
criterion = nn.CrossEntropyLoss()   
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. TRAIN
print("Training...")
for epoch in range(5): 
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()         
        outputs = model(images)        
        loss = criterion(outputs, labels)  
        optimizer.step()              
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/5, Loss: {total_loss/len(train_loader):.4f}")

# 5. TEST
print("\nTesting...")
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")