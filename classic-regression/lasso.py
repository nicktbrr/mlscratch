import numpy as np

class LassoRegression():
    def __init__(self, n_iters, lr=0.001, decay=0.0, bias=True):
        self.n_iters = n_iters
        self.lr = lr
        self.decay = decay
        self.bias = bias

    def fit(self, X: np.array, y: np.array):
        if self.bias:
            X = np.hstack([np.ones(X.shape[0]), X])
        for _ in self.n_iters:
            pass