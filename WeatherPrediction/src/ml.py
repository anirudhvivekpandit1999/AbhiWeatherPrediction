import numpy as np

class AdvancedNN:
    def __init__(self, in_dim, h1, h2, h3, output_dim=1):
        self.W1 = np.random.randn(in_dim, h1) * np.sqrt(2/in_dim)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2/h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, h3) * np.sqrt(2/h2)
        self.b3 = np.zeros(h3)
        self.W4 = np.random.randn(h3, output_dim) * np.sqrt(2/h3)
        self.b4 = np.zeros(output_dim)

    def relu(self, x): return np.maximum(0, x)
    def relu_deriv(self, x): return (x > 0).astype(float)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.relu(self.Z3)
        self.Z4 = self.A3 @ self.W4 + self.b4
        return self.Z4

    def train(self, X, Y, lr=1e-3, epochs=200, decay=0.995):
        for epoch in range(epochs):
            Z4 = self.forward(X)
            R = Z4 - Y.reshape(-1,1)
            loss = np.mean(R**2)
            # backprop
            dZ4 = 2*R/len(X)
            dW4 = self.A3.T @ dZ4; db4 = dZ4.mean(0)
            dA3 = dZ4 @ self.W4.T
            dZ3 = dA3 * self.relu_deriv(self.Z3)
            dW3 = self.A2.T @ dZ3; db3 = dZ3.mean(0)
            dA2 = dZ3 @ self.W3.T
            dZ2 = dA2 * self.relu_deriv(self.Z2)
            dW2 = self.A1.T @ dZ2; db2 = dZ2.mean(0)
            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * self.relu_deriv(self.Z1)
            dW1 = X.T @ dZ1; db1 = dZ1.mean(0)
            # update
            self.W4 -= lr*dW4; self.b4 -= lr*db4
            self.W3 -= lr*dW3; self.b3 -= lr*db3
            self.W2 -= lr*dW2; self.b2 -= lr*db2
            self.W1 -= lr*dW1; self.b1 -= lr*db1
            lr *= decay
            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d}: Loss={loss:.5f}")
