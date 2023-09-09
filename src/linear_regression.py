import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, data, X_map_func=None):
        self.a = None
        self.b = None

        self.y_pred = None

        # Sort dataset by X values
        sorted_indices = np.argsort(data[:, 0])
        self.X = data[:, 0][sorted_indices]
        self.y_true = data[:, 1][sorted_indices]

        # Construct data set S'
        self.X_prime = X_map_func(self.X) if X_map_func else self.X
        self.y_prime = np.log(self.y_true)

    def fit_model(self):
        # Append a column of ones to X
        X_tilde = np.c_[np.ones(self.X_prime.shape[0]), self.X_prime]

        # Calculate (X^T * X)^-1 * X^T * y and obtain parameters a and b
        self.b, self.a = np.linalg.inv(X_tilde.T.dot(X_tilde)).dot(X_tilde.T).dot(self.y_prime)

        # Fit model h'(x) = a * x + b
        h_prime = self.a * self.X_prime + self.b

        # Obtain final model h(x) = exp(h'(x)) (i.e. predictions)
        self.y_pred = np.exp(h_prime)

    def mse(self):
        # MSE = 1/n * sum((y - h(x))^2)
        return np.mean((self.y_true - self.y_pred) ** 2)

    def r2(self):
        # R^2 = 1 - sum((y - h(x))^2) / sum((y - y_bar)^2)
        y_bar = np.mean(self.y_true)
        return 1 - (np.sum((self.y_true - self.y_pred) ** 2) / np.sum((self.y_true - y_bar) ** 2))

    def report(self):
        print(f"a: {self.a}")
        print(f"b: {self.b}")
        print(f"MSE: {self.mse()}")
        print(f"R^2: {self.r2()}")

    def plot(self):
        # Always use "normal" X scale
        plt.scatter(self.X, self.y_true, color="b", label="True values")
        plt.scatter(self.X, self.y_pred, color="r", marker="x", label="Predictions")

        plt.xlabel("Age of the fish")
        plt.ylabel("Log of PCB concentration")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.show()
