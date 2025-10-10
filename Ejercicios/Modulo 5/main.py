# Modulo 5

"""

1. Load dataset
2. Define the model
3. Define the loss function
4. Define the optimizer
5. Train the model
6. Evaluate the model


Dataset: book or corpus of text
Preprocessing: tokenization, vocabularies, embeddings
Models: LSTM/GRU stacked + fc
Train: prediction of netx token and generate text
Evaluation: perplexity, visualization
Optional: regularization (dropout, gradient clipping, early stopping, scheduler), teacher forcing



Estructura:
Proyecto_Texto/
├── train.py
├── generate.py
├── data/
│   └── corpus.txt
├── models/
│   └── lstm_model.py  (opcional si modularizas)
├── outputs/
│   ├── loss_curve.png
│   ├── generated_samples.txt
│   └── best_model.pth

"""

# train.py

# Imports

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torch.cuda.amp import autocast, GradScaler

from torch.nn.utils import clip_grad_norm_

import os


# Dataset
script_dir = os.path.dirname(os.path.abspath(__file__))  # directorio donde está main.py
file_path = os.path.join(script_dir, "data", "corpus.txt")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read().lower()

# Tokenization
tokens = text.split()  # divide por espacios

vocab = sorted(set(tokens))
token_to_idx = {token: idx for idx, token in enumerate(vocab)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}

encoded = [token_to_idx[token] for token in tokens]


seq_len = 30
data = []

for i in range(len(encoded) - seq_len):
    x = torch.tensor(encoded[i:i+seq_len])      # entrada
    y = torch.tensor(encoded[i+seq_len])        # siguiente token
    data.append((x, y))

train_loader = DataLoader(data, batch_size=64, shuffle=True)


# Model 
class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout= 0.3)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x) 
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# Train
def train(model, train_loader, optimizer, criterion, device, max_grad_norm):
    model.train()
    scaler = GradScaler()  # inicializamos fuera del bucle de épocas
    
    for epoch in range(10):
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # --- entrenamiento en precisión mixta ---
            with autocast():  
                y_pred = model(x)
                loss = criterion(y_pred, y)

            # --- backward con escalado ---
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)  # desescala los gradientes antes de clip
            clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Época {epoch+1} | Pérdida promedio: {total_loss/len(train_loader):.4f}")


# Evaluate
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()

    return total_loss / len(val_loader)


# Split data
train_size = int(0.8 * len(data))
val_size = len(data) - train_size

train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)


# define the model
model = LSTMNet(vocab_size=len(vocab), embedding_dim=128, hidden_size=256, output_size=len(vocab))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


best_loss = float('inf')

for epoch in range(10):
    train(model, train_loader, optimizer, criterion, device, 1.0)

    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")

    # Guardar el mejor modelo
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("✅ Mejor modelo guardado")

