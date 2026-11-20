import numpy as np


def relu(z):
    return np.maximum(0.0,z)

def d_relu(z):
    return (z > 0.0).astype(np.float64)

# generate data :
rng = np.random.default_rng(0)
N = 300
X = rng.uniform(-np.pi, np.pi, size=(N, 1))
y = np.sin(X)

# weights (1 and 2) and bias  :
hidden_dim = 16

W1 = rng.normal(0.0, 0.5, size=(1, hidden_dim))
b1 = np.zeros((1, hidden_dim))
W2 = rng.normal(0.0, 0.5, size=(hidden_dim, 1))
b2 = np.zeros((1, 1))

lr = 0.05
epoches = 1000

for epoch in range(epoches):
    
    z1 = X @ W1 + b1
    a1 = relu(z1)
    y_pred = a1 @ W2 + b2
    loss = np.mean((y_pred-y)**2)

    error = y_pred - y
    loss = np.mean(error ** 2) # MSE 
    
    dy_pred= (2.0/N) * error
    dW2 = a1.T @ dy_pred
    db2 = np.sum(dy_pred,axis=0,keepdims=True) 
 
    da1 = dy_pred @ W2.T 
    dz1 = da1 * d_relu(z1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1,axis=0,keepdims=True)
    
    # update grediants :
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1 
    b1 -= lr * db1

    if(epoch+1)%60==0:
        print(f"""Epoch {epoch+1:3d} 
            | Loss:{loss:.4f} 
            | W1: {W1[0,0]:.3f} 
            | b1 :{b1[0,0]:.3f}
            | W2 :{W2[0,0]:.3f}
            | b2 : {b2[0,0]:.3f}"""
              )
