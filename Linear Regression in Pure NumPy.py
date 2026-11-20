import numpy as np

print(np.__version__)

# generate data :
rng = np.random.default_rng(42)
N = 200
X = rng.uniform(-2.0, 2.0, size=(N, 1))
noise = rng.normal(0.0, 0.25, size=(N, 1))
y = 3.0 * X + 7.0 + noise

# weights and bias :
W = rng.normal(0.0, 0.1, size=(1, 1))
b = np.zeros((1, 1))

lr = 0.05
epoches = 300


for epoch in range(epoches):
    y_pred = X @ W + b
    error = y_pred - y
    loss = np.mean(error ** 2) # MSE 
    # backwarde pass 
    # d/d(y_pred) [(y_pred - y)^2 / N] = 2 * (y_pred - y) / N
    d_loss_dy_pred= (2.0/N) * error
    dW = X.T @ d_loss_dy_pred
    db = np.sum(d_loss_dy_pred,axis=0,keepdims=True) # shape (1,1)

    # update grediants :
    W -= lr * dW
    b -= lr * db

    if(epoch+1)%60==0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | W: {W[0,0]:.3f} | b: {b[0,0]:.3f}")


print(f"formula {W[0,0]:.2f} * X + {b[0,0]:.2f}")
print(f"{abs((3 - W[0,0])):.2f}%")
