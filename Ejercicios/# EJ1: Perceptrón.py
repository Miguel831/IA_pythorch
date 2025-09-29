# EJ1: Perceptrón


#Implementation with numpy

import numpy as np

def sigmoid(x):  # Conviarte a valor entre 0-1, segun la probabuilidad de que la clase sea 1
    return 1 / (1 + np.exp(-x))

def train_logistic_regression(X, y, learning_rate, num_epochs):
    n,d = X.shape  # [ ejemplos, caracteristicas ]
    w = np.zeros(d)     # Matriz de pesos 
    b = 0   # Bias

    for _ in range(num_epochs):
        y_pred = sigmoid(np.dot(X, w) + b) # Pasamos el producto vectorial entre la matriz de entrada y la matriz de pesos mas bias -> sigmoid
        dw = np.dot(X.T, (y_pred - y)) / n  # Producto vectorial entre la matriz de entrada transpuesta y el vector de errores y luego lo normaliza
        db = np.sum(y_pred - y) / n  # Suma los errores entre preduccion y fin y divide entre el total = error medio

        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b


# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

learning_rate = 0.1
num_epochs = 1000

w, b = train_logistic_regression(X, y, learning_rate, num_epochs)

print("Weights:", w)
print("Bias:", b)



# Implementacion con torch

import torch
import torch.nn as nn
import torch.optim as optim

def train_logistic_regression_torch(X, y, learning_rate, num_epochs):
    n,d = X.shape
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    model = nn.Sequential(
        nn.Linear(d, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.SGD([w, b], lr=learning_rate)

    for _ in range(num_epochs):
        y_pred = model(X )
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return w.detach().numpy(), b.detach().numpy()


#Example usage
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

learning_rate = 0.1
num_epochs = 1000

w, b = train_logistic_regression_torch(X, y, learning_rate, num_epochs)

print("Weights:", w)
print("Bias:", b)