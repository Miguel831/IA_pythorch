# train_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_cifar10_loaders(batch_size=64, num_workers=4, data_dir='./data'):
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

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=train_transforms, download=True)
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=val_transforms, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_dataset.classes

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in tqdm(loader, desc='Train', leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast(enabled=(scaler is not None)):
            outputs = model(data)
            loss = criterion(outputs, target)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * data.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in tqdm(loader, desc='Val', leave=False):
        data, target = data.to(device), target.to(device)
        with autocast(enabled=(scaler is not None)):
            outputs = model(data)
            loss = criterion(outputs, target)
        running_loss += loss.item() * data.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def save_checkpoint(state, filename='best_model.pth'):
    torch.save(state, filename)

def plot_curves(logs, out_path='training_curves.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(logs['train_loss'], label='train_loss')
    plt.plot(logs['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(logs['train_acc'], label='train_acc')
    plt.plot(logs['val_acc'], label='val_acc')
    plt.legend(); plt.title('Accuracy')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

@torch.no_grad()
def get_predictions_and_labels(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for data, target in tqdm(loader, desc='Predict', leave=False):
        data = data.to(device)
        outputs = model(data)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(target.numpy().tolist())
    return all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, class_names, out_path='confusion.png'):
    cm = confusion_matrix(y_true, y_pred)
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def print_classification_report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names))

def set_parameter_requires_grad(model, feature_extracting: bool):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
