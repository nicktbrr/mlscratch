import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000, bias=True):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = bias
        self.N = None
        self.losses = []

    def fit(self, X: np.array, y: np.array) -> bool:
        if self.bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        if not self.N:
            self.N = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.n_iters):
            y_hat = np.dot(X, self.weights)
            loss = self._MSE(y, y_hat)
            self.losses.append(loss)
            error = y - y_hat
            grad = np.dot(X.T, error) / self.N

            self.weights -= self.learning_rate * grad

        return True
            


    def predict(self, X: np.array):
        pass

    def _MSE(self, y: np.array, y_hat: np.array) -> np.float64:
        return np.sum(np.pow(y - y_hat, 2))


lin = LinearRegression()
X = np.random.rand(3,2)
y = np.random.rand(3)

print(X.shape, X)
print(y.shape, y)

lin.fit(X, y)

print(type(lin._MSE(y,y)))