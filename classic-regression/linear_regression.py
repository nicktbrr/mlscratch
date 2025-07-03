import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as SKLinearRegression

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
        self.N = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.n_iters):
            y_hat = np.dot(X, self.weights)
            loss = self._MSE(y, y_hat)
            self.losses.append(loss)
            error = y_hat - y
            grad = np.dot(X.T, error) / self.N

            self.weights -= self.learning_rate * grad

        return True

    def predict(self, X: np.array):
        if self.bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.dot(X, self.weights)

    def _MSE(self, y: np.array, y_hat: np.array) -> np.float64:
        return np.sum(np.power(y - y_hat, 2)) / self.N
    
def main():
    print("--- Starting Linear Regression Tests ---")

    # Test 1: Simple 1D Linear Regression
    print("\n--- Test 1: Simple 1D Linear Regression ---")
    # Generate synthetic data: y = 2x + 5 + noise
    X = np.random.rand(100, 1) * 10 # 100 samples, 1 feature, values from 0 to 10
    y = 2 * X + 5 + np.random.randn(100, 1) * 2 # y = 2x + 5 + some noise

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Flatten y arrays for consistent shape (e.g., (N,) instead of (N,1))
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Create and train our custom Linear Regression model
    my_lr_model = LinearRegression(learning_rate=0.01, n_iters=1000)
    print(f"Fitting custom Linear Regression model (learning_rate={my_lr_model.learning_rate}, n_iters={my_lr_model.n_iters})...")
    my_lr_model.fit(X_train, y_train)
    print("Fitting complete.")

    # Make predictions
    my_predictions = my_lr_model.predict(X_test)

    # Evaluate MSE
    my_mse = my_lr_model._MSE(y_test, my_predictions)
    print(f"Custom Linear Regression MSE: {my_mse:.4f}")
    # Print learned weights (bias, then feature coefficients)
    print(f"Custom Linear Regression Weights (Bias, Coef): {my_lr_model.weights}")

    # Compare with Scikit-learn's Linear Regression
    sk_lr_model = SKLinearRegression()
    sk_lr_model.fit(X_train, y_train)
    sk_predictions = sk_lr_model.predict(X_test)
    sk_mse = mean_squared_error(y_test, sk_predictions)
    print(f"Scikit-learn Linear Regression MSE: {sk_mse:.4f}")
    print(f"Scikit-learn Intercept: {sk_lr_model.intercept_:.4f}")
    print(f"Scikit-learn Coefficient: {sk_lr_model.coef_[0]:.4f}")

    # Test 2: Multiple Features (2D) Linear Regression
    print("\n--- Test 2: Multiple Features (2D) Linear Regression ---")
    # Generate synthetic data: y = 1.5x1 + 3x2 + 10 + noise
    X_multi = np.random.rand(100, 2) * 10 # 100 samples, 2 features
    y_multi = 1.5 * X_multi[:, 0] + 3 * X_multi[:, 1] + 10 + np.random.randn(100) * 3

    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

    my_lr_model_multi = LinearRegression(learning_rate=0.005, n_iters=2000, bias=True)
    print(f"Fitting custom Multi-feature LR model (learning_rate={my_lr_model_multi.learning_rate}, n_iters={my_lr_model_multi.n_iters})...")
    my_lr_model_multi.fit(X_train_multi, y_train_multi)
    print("Fitting complete.")

    my_predictions_multi = my_lr_model_multi.predict(X_test_multi)
    my_mse_multi = my_lr_model_multi._MSE(y_test_multi, my_predictions_multi)
    print(f"Custom Multi-feature LR MSE: {my_mse_multi:.4f}")
    print(f"Custom Multi-feature LR Weights (Bias, Coef1, Coef2): {my_lr_model_multi.weights}")

    sk_lr_model_multi = SKLinearRegression()
    sk_lr_model_multi.fit(X_train_multi, y_train_multi)
    sk_predictions_multi = sk_lr_model_multi.predict(X_test_multi)
    sk_mse_multi = mean_squared_error(y_test_multi, sk_predictions_multi)
    print(f"Scikit-learn Multi-feature LR MSE: {sk_mse_multi:.4f}")
    print(f"Scikit-learn Multi-feature Intercept: {sk_lr_model_multi.intercept_:.4f}")
    print(f"Scikit-learn Multi-feature Coefficients: {sk_lr_model_multi.coef_}")
    
    print("\n--- All Tests Completed ---")

if __name__ == "__main__":
    main()