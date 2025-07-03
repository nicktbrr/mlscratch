import numpy as np

class LassoRegressionScratch:
    def __init__(self, alpha=1.0, n_iters=1000, tol=1e-4):
        self.alpha = alpha
        self.n_iters = n_iters
        self.tol = tol
        self.weights = None
        self.bias = 0.0

    def _soft_threshold(self, rho, alpha_val):
        if rho > alpha_val:
            return rho - alpha_val
        elif rho < -alpha_val:
            return rho + alpha_val
        else:
            return 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.y_mean = np.mean(y)

        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)
        y_centered = y - self.y_mean

        self.weights = np.zeros(n_features)

        feature_l2_norms = np.sum(X_scaled**2, axis=0)

        for iteration in range(self.n_iters):
            weights_old = np.copy(self.weights)

            for j in range(n_features):
                y_pred_excluding_j = X_scaled @ self.weights - X_scaled[:, j] * self.weights[j]
                rho_j = np.dot(X_scaled[:, j], y_centered - y_pred_excluding_j)

                self.weights[j] = self._soft_threshold(rho_j, self.alpha * n_samples / 2)

                if feature_l2_norms[j] > 1e-8:
                    self.weights[j] /= feature_l2_norms[j]

            if np.linalg.norm(self.weights - weights_old) < self.tol:
                break

        self.bias = self.y_mean - np.sum(self.weights * (self.X_mean / (self.X_std + 1e-8)))


    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)
        return X_scaled @ self.weights + self.bias

if __name__ == "__main__":
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    np.random.seed(42)
    X_scratch = np.random.rand(100, 5) * 10
    true_coefs_scratch = np.array([2.5, 0.0, -1.2, 0.0, 3.1])
    y_scratch = X_scratch @ true_coefs_scratch + np.random.randn(100) * 0.5

    X_train_scratch, X_test_scratch, y_train_scratch, y_test_scratch = train_test_split(
        X_scratch, y_scratch, test_size=0.2, random_state=42
    )

    lasso_scratch_model = LassoRegressionScratch(alpha=0.1, n_iters=1000)
    lasso_scratch_model.fit(X_train_scratch, y_train_scratch)

    y_pred_scratch = lasso_scratch_model.predict(X_test_scratch)

    mse_scratch = mean_squared_error(y_test_scratch, y_pred_scratch)
    print(f"\nScratch Implementation MSE: {mse_scratch:.4f}")
    print("Scratch Implementation Coefficients:", lasso_scratch_model.weights)
    print("Scratch Implementation Intercept:", lasso_scratch_model.bias)

    print("\n--- Comparison with scikit-learn ---")
    scaler = StandardScaler()
    X_train_scaled_sk = scaler.fit_transform(X_train_scratch)
    X_test_scaled_sk = scaler.transform(X_test_scratch)

    lasso_sk_model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
    lasso_sk_model.fit(X_train_scaled_sk, y_train_scratch)

    y_pred_sk = lasso_sk_model.predict(X_test_scaled_sk)
    mse_sk = mean_squared_error(y_test_scratch, y_pred_sk)
    print(f"Scikit-learn MSE: {mse_sk:.4f}")
    print("Scikit-learn Coefficients:", lasso_sk_model.coef_)
    print("Scikit-learn Intercept:", lasso_sk_model.intercept_)