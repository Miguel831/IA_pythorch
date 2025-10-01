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

    model = nn.Sequential(
        nn.Linear(d, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for _ in range(num_epochs):
        y_pred = model(X)  # Calculates the output of the model
        loss = criterion(y_pred, y)  # Calculates the loss (grafo de pérdida)
        loss.backward() # Backpropagation -- Calcula el gradiente de la función de pérdida respecto a los parámetros del modelo (backpropagation en el grafo)
        optimizer.step() # Updates the model parameters
        optimizer.zero_grad() # Sets the gradients of the model parameters to zero
        # Early stopping
        if loss < 0.0001:
            break

    return model[0].weight.detach().numpy(), model[0].bias.detach().numpy()

# Example usage
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)  # <- corregido

learning_rate = 0.1
num_epochs = 1000

w, b = train_logistic_regression_torch(X, y, learning_rate, num_epochs)

print("Weights:", w)
print("Bias:", b)