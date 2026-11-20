import torch 
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
#import pandas as pd -- no need for it 

torch.manual_seed(42)
rng = np.random.default_rng(42)

data = load_breast_cancer()
X = data.data.astype(np.float32)
y = data.target.astype(np.float32).reshape(-1, 1)

X_train , X_test , y_train , y_test = train_test_split(X,y)

#X_train , y_train
mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True) + 1e-8
X_train_t = torch.from_numpy((X_train - mean) / std)
y_train_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy((X_test - mean) / std)
y_test_t = torch.from_numpy(y_test)

in_dim = X_train_t.shape[1]
model = nn.Sequential(
    nn.Linear(in_dim, 16),
    nn.Tanh(),
    nn.Linear(16, 1),
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

epochs = 1200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 300 == 0:
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == y_train_t).float().mean().item() * 100.0
            print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.4f} | Train Acc: {acc:.2f}%")

model.eval()
with torch.no_grad():
    test_logits = model(X_test_t)
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs >= 0.5).float()
    test_acc = (test_preds == y_test_t).float().mean().item() * 100.0

print(f"\nUnseen Test Accuracy ({len(X_test_t)} samples): {test_acc:.2f}%")


