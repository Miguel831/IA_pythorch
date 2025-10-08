# Train.py

"""

Entrenar desde 0.
Feature extraction
Fine-tuning

Steps:
1. Load data
2. Define the model - ResNet
3. Define the loss function
4. Define the optimizer
5. Train the model
6. Evaluate the model


1. Load data - Transforms & Dataloader   (CIFAR-10)
2. Load resnet18(pretrained=True) - Train head 
3. Unfreeze layers and continue training
4. Compare losses and accuracy - Confusion matrix
5. Save best model

"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torch.cuda.amp import autocast, GradScaler

from torchvision import datasets,transforms
from torchvision.models import resnet18, ResNet18_Weights



# Transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


# Dataloader

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    transform=train_transforms,   # se aplica a cada imagen
    download=True
)  

val_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    transform=val_transforms,
    download=True
)  


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)  # modelo pre-entrenado, sin head

# Train head
model.fc = nn.Linear(in_features=512, out_features=10)  # head, nueva capa

model = model.to(device)




# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


# Train head
scaler = GradScaler()
for epoch in range(3):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


    # Evaluate the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader)

    accuracy = 100. * correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        accuracy))
    


