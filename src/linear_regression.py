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

        # Construct the data set S'
        self.X_prime = X_map_func(self.X) if X_map_func else self.X
        self.y_prime = np.log(self.y_true)

    def fit_model(self):
        # Append a column of ones to X
        X_tilde = np.c_[np.ones(self.X_prime.shape[0]), self.X_prime]

        # Calculate w = (X^T * X)^-1 * X^T * y (normal equation)
        w = np.linalg.inv(X_tilde.T.dot(X_tilde)).dot(X_tilde.T).dot(self.y_prime)

        # Extract parameters a and b
        self.b, self.a = w

        # Fit the model
        h_prime = self.a * self.X_prime + self.b

        # Obtain final model h(x)=exp(h_prime(x)) (i.e. predictions)
        self.y_pred = np.exp(h_prime)

    def mean_squared_error(self):
        return np.mean((self.y_true - self.y_pred) ** 2)

    def r_squared(self):
        y_bar = np.mean(self.y_true)

        return 1 - (np.sum((self.y_true - self.y_pred) ** 2) / np.sum((self.y_true - y_bar) ** 2))

    def report(self):
        print(f"Parameter a: {self.a}")
        print(f"Parameter b: {self.b}")
        print(f"Mean squared error: {self.mean_squared_error()}")
        print(f"R^2: {self.r_squared()}")

    def plot(self):
        # always use "normal" X scale
        plt.scatter(self.X, self.y_true, color="b", label="True values")
        plt.scatter(self.X, self.y_pred, color="r", marker="x", label="Predictions")

        plt.xlabel("Age of the fish")
        plt.ylabel("PCB concentration")
        plt.legend()
        plt.yscale("log")
        plt.show()
