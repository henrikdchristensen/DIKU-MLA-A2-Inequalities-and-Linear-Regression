import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, X, y_true, transform_X=False, transform_y=False):
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.a = None
        self.b = None
        self.X_transformed = None
        self.y_transformed = None
        self.y_pred = None

        # sort the dataset by X
        sorted_indices = np.argsort(X, axis=0).reshape(-1)
        self.X = X[sorted_indices]
        self.y_true = y_true[sorted_indices]

    def _construct_dataset(self):
        self.X_transformed = np.sqrt(self.X) if self.transform_X else self.X
        self.y_transformed = np.log(self.y_true) if self.transform_y else self.y_true

    def fit_model(self):
        self._construct_dataset()

        X_b = np.c_[np.ones((self.X_transformed.shape[0], 1)), self.X_transformed]
        weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(self.y_transformed)
        self.b, self.a = weights

    def predict(self):
        h_prime = self.a * self.X_transformed + self.b
        self.y_pred = np.exp(h_prime) if self.transform_y else h_prime

    def mean_squared_error(self):
        return np.mean((self.y_true - self.y_pred) ** 2)

    def r_squared(self):
        residual_sum_of_squares = np.sum((self.y_true - self.y_pred) ** 2)
        total_sum_of_squares = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        return 1 - (residual_sum_of_squares / total_sum_of_squares)

    def report(self):
        print(f"Parameter a: {self.a}")
        print(f"Parameter b: {self.b}")
        print(f"Mean squared error: {self.mean_squared_error()}")
        print(f"R^2: {self.r_squared()}")
        # print predictions and true values

    def plot(self):
        # plot the true values and predictions
        # always use "normal" X scale
        plt.scatter(self.X, self.y_true, color="b", label="True values")
        plt.scatter(self.X, self.y_pred, color="r", marker="x", label="Predictions")

        plt.xlabel("Age of the fish")
        plt.ylabel("PCB concentration")
        plt.legend()
        plt.yscale("log")
        plt.show()


# Sample usage:
