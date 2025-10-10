# models/resnet_finetune.py
"""Fine-tuning parcial: partir del modelo feature-extract y descongelar layer4 + fc"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from train_utils import seed_everything, get_cifar10_loaders, train_one_epoch, evaluate, save_checkpoint, plot_curves, get_predictions_and_labels, plot_confusion_matrix, print_classification_report
from torch.cuda.amp import GradScaler
import os

def main():
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, class_names = get_cifar10_loaders(batch_size=64)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 10)

    fe_path = 'best_resnet_fe.pth'
    if os.path.exists(fe_path):
        print('Cargando pesos de feature extract:', fe_path)
        ckpt = torch.load(fe_path)
        model.load_state_dict(ckpt['model_state'])

    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    scaler = GradScaler()
    best_acc = 0.0
    logs = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    epochs = 20
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, scaler)
        scheduler.step()

        logs['train_loss'].append(train_loss)
        logs['val_loss'].append(val_loss)
        logs['train_acc'].append(train_acc)
        logs['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({'epoch': epoch+1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'val_acc': val_acc}, filename='best_resnet_finetune.pth')

    plot_curves(logs, out_path='resnet_finetune_curves.png')

    model.eval()
    y_pred, y_true = get_predictions_and_labels(model, val_loader, device)
    plot_confusion_matrix(y_true, y_pred, class_names, out_path='resnet_finetune_confusion.png')
    print_classification_report(y_true, y_pred, class_names)

if __name__ == '__main__':
    main()
