import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from collections import Counter
import math

# ============================================================================
# 1. PREPROCESAMIENTO Y DATASET
# ============================================================================

class TextPreprocessor:
    """Tokenización, construcción de vocabulario y conversión a índices"""
    
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def tokenize(self, text):
        """Tokenización básica: limpieza y split por palabras"""
        # Convertir a minúsculas y limpiar
        text = text.lower()
        # Mantener puntuación importante
        text = re.sub(r'[^a-záéíóúñü\s\.\,\;\:\!\?\-]', '', text)
        # Tokenizar por palabras y puntuación
        tokens = re.findall(r'\w+|[.,;:!?]', text)
        return tokens
    
    def build_vocab(self, text):
        """Construir vocabulario con tokens frecuentes"""
        tokens = self.tokenize(text)
        counter = Counter(tokens)
        
        # Tokens especiales
        self.word2idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        
        # Añadir palabras frecuentes
        idx = 4
        for word, freq in counter.most_common():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        print(f"Vocabulario construido: {self.vocab_size} tokens")
        print(f"Tokens totales en texto: {len(tokens)}")
        
    def encode(self, text):
        """Convertir texto a secuencia de índices"""
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
    
    def decode(self, indices):
        """Convertir índices a texto"""
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])


class TextDataset(Dataset):
    """Dataset para secuencias de texto"""
    
    def __init__(self, encoded_text, seq_len):
        self.data = encoded_text
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


# ============================================================================
# 2. MODELO LSTM/GRU
# ============================================================================

class TextGeneratorLSTM(nn.Module):
    """Modelo LSTM stacked para generación de texto"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        
        output = self.dropout(output)  # (batch, seq_len, hidden_size)
        logits = self.fc(output)       # (batch, seq_len, vocab_size)
        
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        """Inicializar hidden y cell states"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


# ============================================================================
# 3. ENTRENAMIENTO
# ============================================================================

def calculate_perplexity(loss):
    """Calcular perplexity desde loss"""
    return math.exp(min(loss, 100))  # Limitar para evitar overflow


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
    """Entrenar una época"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(x)  # (batch, seq_len, vocab_size)
        
        # Reshape para calcular loss
        logits_flat = logits.view(-1, logits.size(-1))  # (batch*seq_len, vocab_size)
        y_flat = y.view(-1)  # (batch*seq_len)
        
        loss = criterion(logits_flat, y_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # Métricas
        total_loss += loss.item()
        
        # Accuracy
        predictions = logits_flat.argmax(dim=1)
        total_correct += (predictions == y_flat).sum().item()
        total_tokens += y_flat.size(0)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * total_correct / total_tokens
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, accuracy, perplexity


def evaluate(model, dataloader, criterion, device):
    """Evaluar el modelo"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            logits, _ = model(x)
            
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            
            loss = criterion(logits_flat, y_flat)
            total_loss += loss.item()
            
            predictions = logits_flat.argmax(dim=1)
            total_correct += (predictions == y_flat).sum().item()
            total_tokens += y_flat.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * total_correct / total_tokens
    perplexity = calculate_perplexity(avg_loss)
    
    return avg_loss, accuracy, perplexity


# ============================================================================
# 4. MAIN - ENTRENAMIENTO COMPLETO
# ============================================================================

def main():
    # Configuración
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {DEVICE}")
    
    # Hiperparámetros
    SEQ_LEN = 50
    BATCH_SIZE = 64
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3
    DROPOUT = 0.4
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    CLIP_GRAD = 1.0
    
    # Cargar texto (simulado aquí - debes cargar el Quijote real)
    print("Cargando texto del Quijote...")
    
    # IMPORTANTE: Aquí debes cargar el texto real del Quijote
    # Ejemplo: with open('quijote.txt', 'r', encoding='utf-8') as f: text = f.read()
    
    # Para esta demostración, usaré un fragmento simulado
    texto_quijote = """
    En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía 
    un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. 
    Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados,
    lantejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda.
    """ * 100  # Repetir para tener más datos
    
    # Preprocesamiento
    print("\nPreprocesando texto...")
    preprocessor = TextPreprocessor(min_freq=2)
    preprocessor.build_vocab(texto_quijote)
    encoded_text = preprocessor.encode(texto_quijote)
    
    # Split train/val
    split_idx = int(0.9 * len(encoded_text))
    train_data = encoded_text[:split_idx]
    val_data = encoded_text[split_idx:]
    
    # Datasets y dataloaders
    train_dataset = TextDataset(train_data, SEQ_LEN)
    val_dataset = TextDataset(val_data, SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Datos de entrenamiento: {len(train_dataset)} secuencias")
    print(f"Datos de validación: {len(val_dataset)} secuencias")
    
    # Modelo
    model = TextGeneratorLSTM(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    print(f"\nModelo creado con {sum(p.numel() for p in model.parameters())} parámetros")
    
    # Optimización
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignorar padding
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Entrenamiento
    train_losses, train_accs, train_perps = [], [], []
    val_losses, val_accs, val_perps = [], [], []
    best_val_loss = float('inf')
    
    print("\nIniciando entrenamiento...\n")
    
    for epoch in range(NUM_EPOCHS):
        print(f"Época {epoch + 1}/{NUM_EPOCHS}")
        
        # Entrenar
        train_loss, train_acc, train_perp = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, CLIP_GRAD
        )
        
        # Evaluar
        val_loss, val_acc, val_perp = evaluate(model, val_loader, criterion, DEVICE)
        
        # Guardar métricas
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_perps.append(train_perp)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_perps.append(val_perp)
        
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Perplexity: {train_perp:.2f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Perplexity: {val_perp:.2f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, 'best_model.pt')
            print(f"  ✓ Mejor modelo guardado (val_acc: {val_acc:.2f}%)")
        
        print()
    
    # Guardar preprocessor y configuración
    config = {
        'vocab_size': preprocessor.vocab_size,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'seq_len': SEQ_LEN,
        'word2idx': preprocessor.word2idx,
        'idx2word': preprocessor.idx2word
    }
    
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("Configuración guardada en config.json")
    
    # Visualización de resultados
    plt.figure(figsize=(15, 5))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Evolución del Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Época')
    plt.ylabel('Accuracy (%)')
    plt.title('Evolución de la Precisión')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=80, color='r', linestyle='--', label='Objetivo 80%')
    
    # Perplexity
    plt.subplot(1, 3, 3)
    plt.plot(train_perps, label='Train Perplexity')
    plt.plot(val_perps, label='Val Perplexity')
    plt.xlabel('Época')
    plt.ylabel('Perplexity')
    plt.title('Evolución de Perplexity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    print("Gráficos guardados en training_metrics.png")
    
    print(f"\n✓ Entrenamiento completado!")
    print(f"Mejor precisión en validación: {max(val_accs):.2f}%")


if __name__ == '__main__':
    main()