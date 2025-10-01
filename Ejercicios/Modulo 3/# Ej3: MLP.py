# Ej3: MLP.py

"""
1. Load data
2. Define the model
3. Define the loss function
4. Define the optimizer
5. Train the model
6. Evaluate the model

Requirements:
- 2 - 4 hidden layers -
- BachNormalization -
- Dropout - 
- Esarly stopping
- Save best model
- Tensorboard logs
- Inicialize weights

Compare optimizers:
- AdamW
- SGD with momentum

Compare regularization:
- weight decay
- dropout

LR schedule:
- ReduceLROnPlateau
- OneCycleLR 

"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torch.cuda.amp import autocast, GradScaler



# Loggin in tensorboard

from torch.utils.tensorboard import SummaryWriter
import os
import shutil


# Load the dataset

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Loggin in tensorboard

from torch.utils.tensorboard import SummaryWriter
import os
import shutil


# Load data --> FashionMNIST

train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
val_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor())
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the model

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),  #normaliza las activaciones de una capa usando la media y desviación estándar del batch actual.
            nn.ReLU(),
            nn.Dropout(0.2), # apaga aleatoriamente un porcentaje p de neuronas
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
    

# Initialize the model

model = MLP()


# Define the loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)


# Train the model

scaler = GradScaler()
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with autocast():  # Automatic Mixed Precision
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Logging
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("Loss/train", loss.item(), global_step)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]['lr'], global_step)

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            

# Evaluate the model

def evaluate(model, test_loader, criterion, device, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    accuracy = 100. * correct / len(test_loader.dataset)

    # Logging
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", accuracy, epoch)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    
# Save model

def save_model(model, path, best_accuracy):
    torch.save({
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'best_accuracy': best_accuracy
}, path)


# Early stopping

class EarlyStopping:
    def __init__(self, patience=3, verbose=True):
        self.patience = patience      # número de épocas sin mejora antes de parar
        self.verbose = verbose
        self.best_accuracy = 0
        self.counter = 0
        self.early_stop = False

    def __call__(self, accuracy, model):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.counter = 0
            # Guardar el mejor modelo automáticamente
            torch.save(model.state_dict(), 'best_model.pth')
            if self.verbose:
                print(f"Mejora detectada: {accuracy:.2f}%, modelo guardado.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No mejora en {self.counter} época(s) consecutivas.")
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping activado por falta de mejora.")
                self.early_stop = True
