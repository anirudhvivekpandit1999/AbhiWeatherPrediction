import numpy as np

class SimpleLSTM:
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # LSTM weights (simplified, not for production)
        self.Wf = np.random.randn(input_dim + hidden_dim, hidden_dim)
        self.bf = np.zeros(hidden_dim)
        self.Wi = np.random.randn(input_dim + hidden_dim, hidden_dim)
        self.bi = np.zeros(hidden_dim)
        self.Wc = np.random.randn(input_dim + hidden_dim, hidden_dim)
        self.bc = np.zeros(hidden_dim)
        self.Wo = np.random.randn(input_dim + hidden_dim, hidden_dim)
        self.bo = np.zeros(hidden_dim)
        self.Wy = np.random.randn(hidden_dim, output_dim)
        self.by = np.zeros(output_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        # X shape: (batch, seq_len, input_dim)
        batch, seq_len, _ = X.shape
        h = np.zeros((batch, self.hidden_dim))
        c = np.zeros((batch, self.hidden_dim))
        for t in range(seq_len):
            xh = np.concatenate([X[:, t, :], h], axis=1)
            f = self.sigmoid(xh @ self.Wf + self.bf)
            i = self.sigmoid(xh @ self.Wi + self.bi)
            o = self.sigmoid(xh @ self.Wo + self.bo)
            c_tilde = np.tanh(xh @ self.Wc + self.bc)
            c = f * c + i * c_tilde
            h = o * np.tanh(c)
        y = h @ self.Wy + self.by
        return y

    def train(self, X, Y, lr=1e-3, epochs=100):
        # Dummy train: not implemented for brevity
        print("[SimpleLSTM] Training is a placeholder. Use a real LSTM for production.")
