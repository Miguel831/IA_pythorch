# Ej2: CNN.py
"""
1. Load the dataset
2. Define the model
3. Define the loss function
4. Define the optimizer
5. Train the model
6. Evaluate the model

Goals:
- 98% accuracy - 

Requirements:
- Early stopping - 
- Save best model - 
- Logging in tensorboard -
- AMP - 

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



transform = transforms.Compose([
    transforms.ToTensor(),                    # convierte a tensor en [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # mean, std para MNIST
])

train_ds = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)



# 2. Define the model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers -> image recognition
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (1 canal → 32 mapas)
            nn.ReLU(),
            nn.MaxPool2d(2),  # reduce 28x28 → 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (32 → 64 mapas)
            nn.ReLU(),
            nn.MaxPool2d(2)   # reduce 14x14 → 7x7
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # a vector
            nn.Linear(64*7*7, 128),  # capa oculta
            nn.ReLU(),
            nn.Linear(128, 10)       # 10 clases de MNIST
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    


#Train the model optimizad with AMP


scaler = GradScaler()
def train(model, train_loader, optimizer, criterion, device, epoch):
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


        # Logging
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("Loss/train", loss.item(), global_step)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]['lr'], global_step)

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# 6. Evaluate the model

def evaluate(model, test_loader, criterion, device, epoch, best_accuracy=0):
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
    
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        save_model(model, 'best_model.pth', best_accuracy)

    if early_stopper is not None:
        early_stopper(accuracy, model)

    return best_accuracy



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



# main

if __name__ == '__main__':

    logdir = 'runs/cnn_mnist'

    if  os.path.exists(logdir):
        shutil.rmtree(logdir)

    writer = SummaryWriter(logdir)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()


    early_stopper = EarlyStopping(patience=3)


    for epoch in range(1, 6):
        train(model, train_loader, optimizer, criterion, device, epoch)
        evaluate(model, test_loader, criterion, device, epoch)

        if early_stopper.early_stop:
            break

    writer.close()



    
