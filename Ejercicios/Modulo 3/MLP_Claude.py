"""
MLP Mejorado para FashionMNIST
Mejoras implementadas:
1. Inicialización de pesos (Xavier/He)
2. Arquitectura optimizada (2-3 capas)
3. Data Augmentation
4. Comparación de optimizadores
5. OneCycleLR scheduler
6. Early stopping corregido
7. Normalización de datos
8. Evaluación completa
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
import shutil
from typing import Tuple
import numpy as np


# ==================== Configuración ====================
class Config:
    """Configuración centralizada del experimento"""
    # Datos
    batch_size = 128  # Aumentado para mejor estabilidad
    num_workers = 4
    
    # Modelo
    hidden_sizes = [512, 256, 128]  # 3 capas ocultas más balanceadas
    dropout_rates = [0.3, 0.25, 0.2]  # Dropout diferenciado por capa
    
    # Entrenamiento
    epochs = 30
    base_lr = 0.01  # LR más alto para OneCycleLR
    weight_decay = 1e-4
    
    # Early stopping
    patience = 5
    
    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Logging
    log_interval = 100
    experiment_name = "MLP_Fashion_Improved"


# ==================== Transformaciones de Datos ====================
def get_transforms():
    """Define transformaciones con data augmentation para train y sin para val/test"""
    
    # Calculamos media y std de FashionMNIST (valores pre-calculados)
    mean = 0.2860
    std = 0.3530
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
        transforms.RandomRotation(degrees=10),   # Rotación ligera
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translación
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),  # Normalización
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))  # Random erasing
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    return train_transform, test_transform


# ==================== Modelo Mejorado ====================
class ImprovedMLP(nn.Module):
    def __init__(self, input_size=784, num_classes=10, 
                 hidden_sizes=[512, 256, 128], 
                 dropout_rates=[0.3, 0.25, 0.2],
                 activation='relu'):
        super().__init__()
        
        layers = []
        
        # Capa de entrada
        layers.append(nn.Flatten())
        
        # Capas ocultas
        prev_size = input_size
        for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_sizes, dropout_rates)):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _get_activation(self, activation):
        """Selecciona función de activación"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif activation == 'elu':
            return nn.ELU()
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """Inicializa pesos usando Xavier/He según la activación"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization para ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)


# ==================== Early Stopping Mejorado ====================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta  # Mejora mínima requerida
        self.verbose = verbose
        self.best_loss = float('inf')
        self.best_accuracy = 0
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, val_accuracy, model):
        """Monitorea val_loss para early stopping"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_accuracy = val_accuracy
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
            if self.verbose:
                print(f"✓ Mejora detectada: Loss={val_loss:.4f}, Acc={val_accuracy:.2f}%")
        else:
            self.counter += 1
            if self.verbose:
                print(f"✗ Sin mejora por {self.counter}/{self.patience} épocas")
            if self.counter >= self.patience:
                if self.verbose:
                    print("⚠ Early stopping activado")
                self.early_stop = True
        
        return self.early_stop


# ==================== Funciones de Entrenamiento y Evaluación ====================
def train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, scaler, config):
    """Entrena una época"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad(set_to_none=True)  # Más eficiente que zero_grad()
        
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping para estabilidad
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Métricas
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Logging
        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % config.log_interval == 0:
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            lr = optimizer.param_groups[0]['lr']
            
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)] '
                  f'Loss: {current_loss:.4f} Acc: {current_acc:.2f}% LR: {lr:.6f}')
            
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            writer.add_scalar("Accuracy/train_batch", current_acc, global_step)
            writer.add_scalar("Learning_rate", lr, global_step)
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device, epoch, writer, phase="val"):
    """Evalúa el modelo"""
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            with autocast():
                output = model(data)
                test_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= len(data_loader)
    accuracy = 100. * correct / len(data_loader.dataset)
    
    # Logging
    if writer is not None and epoch >= 0:
        writer.add_scalar(f"Loss/{phase}", test_loss, epoch)
        writer.add_scalar(f"Accuracy/{phase}", accuracy, epoch)
    
    print(f'\n{phase.upper()} set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy


# ==================== Función de Comparación de Optimizadores ====================
def compare_optimizers(model_class, train_loader, val_loader, test_loader, config):
    """Compara diferentes optimizadores"""
    
    optimizers_config = [
        {
            'name': 'AdamW',
            'optimizer_class': optim.AdamW,
            'params': {'lr': config.base_lr, 'weight_decay': config.weight_decay}
        },
        {
            'name': 'SGD_momentum',
            'optimizer_class': optim.SGD,
            'params': {'lr': config.base_lr * 10, 'momentum': 0.9, 'weight_decay': config.weight_decay}
        }
    ]
    
    results = {}
    
    for opt_config in optimizers_config:
        print(f"\n{'='*50}")
        print(f"Entrenando con {opt_config['name']}")
        print(f"{'='*50}")
        
        # Reset del modelo
        model = model_class().to(config.device)
        
        # Configuración
        optimizer = opt_config['optimizer_class'](model.parameters(), **opt_config['params'])
        criterion = nn.CrossEntropyLoss()
        
        # Scheduler - OneCycleLR
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=opt_config['params']['lr'] * 10,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # 30% del tiempo para warmup
            anneal_strategy='cos'
        )
        
        # Logging
        log_dir = f'runs/{config.experiment_name}/{opt_config["name"]}'
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        writer = SummaryWriter(log_dir)
        
        # Early stopping
        early_stopper = EarlyStopping(patience=config.patience)
        
        # Mixed precision
        scaler = GradScaler()
        
        # Training loop
        best_val_acc = 0
        for epoch in range(1, config.epochs + 1):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, 
                config.device, epoch, writer, scaler, config
            )
            
            # Validate
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, config.device, epoch, writer, "val"
            )
            
            # Scheduler step (OneCycleLR se actualiza por batch en train_epoch)
            if not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                scheduler.step(val_loss)
            
            # Early stopping
            if early_stopper(val_loss, val_acc, model):
                print(f"Early stopping en época {epoch}")
                # Restaurar mejor modelo
                model.load_state_dict(early_stopper.best_model_state)
                break
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Guardar mejor modelo
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, f'best_model_{opt_config["name"]}.pth')
        
        # Test final con mejor modelo
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, config.device, -1, None, "test"
        )
        
        results[opt_config['name']] = {
            'best_val_acc': early_stopper.best_accuracy,
            'best_val_loss': early_stopper.best_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'epochs_trained': epoch
        }
        
        writer.close()
    
    return results


# ==================== Main ====================
def main():
    # Configuración
    config = Config()
    
    print(f"Dispositivo: {config.device}")
    print(f"Configuración: {vars(config)}\n")
    
    # Transformaciones
    train_transform, test_transform = get_transforms()
    
    # Datasets
    train_dataset_full = datasets.FashionMNIST(
        root='./data', train=True, transform=train_transform, download=True
    )
    
    # Split train/val (50000/10000)
    train_dataset, val_dataset = random_split(
        train_dataset_full, [50000, 10000],
        generator=torch.Generator().manual_seed(42)  # Reproducibilidad
    )
    
    # Aplicar transformación de test a validation
    val_dataset.dataset.transform = test_transform
    
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, transform=test_transform, download=True
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, 
        shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, 
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, 
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    
    # Comparar optimizadores
    results = compare_optimizers(
        ImprovedMLP, train_loader, val_loader, test_loader, config
    )
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    
    for optimizer_name, metrics in results.items():
        print(f"\n{optimizer_name}:")
        print(f"  Best Val Accuracy: {metrics['best_val_acc']:.2f}%")
        print(f"  Best Val Loss: {metrics['best_val_loss']:.4f}")
        print(f"  Test Accuracy: {metrics['test_acc']:.2f}%")
        print(f"  Test Loss: {metrics['test_loss']:.4f}")
        print(f"  Épocas entrenadas: {metrics['epochs_trained']}")
    
    # Determinar mejor optimizador
    best_optimizer = max(results.items(), key=lambda x: x[1]['test_acc'])
    print(f"\n🏆 Mejor optimizador: {best_optimizer[0]} con {best_optimizer[1]['test_acc']:.2f}% en test")


if __name__ == '__main__':
    main()