import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, data, X_map_func=None, plot_filename=None):
        self.X_map_func = X_map_func

        self.plot_filename = plot_filename

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

        self.title = (
            f"Linear Regression of Lake trouts age vs. PCB concentration (ppm)\nx → {f'{self.X_map_func.__name__}(x)' if self.X_map_func else 'x'}"
        )

    def fit_model(self):
        # Append a column of ones to X (equation 3.3 in [CI])
        X_tilde = np.c_[np.ones(self.X_prime.shape[0]), self.X_prime]

        # Calculate w* = (X^T * X)^-1 * X^T * y and obtain parameters a and b (equation 3.5 in [CI])
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
        print("-" * 50)
        print(self.title)
        print(f"a: {self.a:.4f}")
        print(f"b: {self.b:.4f}")
        print(f"MSE: {self.mse():.4f}")
        print(f"R^2: {self.r2():.4f}")
        print("-" * 50)

    def plot(self):
        plt.figure(figsize=(10, 6))
        # Always use "normal" X scale
        plt.scatter(self.X, self.y_true, marker="o", label="True values")
        plt.plot(self.X, self.y_pred, marker="x", linestyle="-", color="r", label="Predictions")
        plt.xlabel("Age of the lake trout (years)")
        plt.xticks(np.arange(min(self.X), max(self.X) + 1, 1))
        plt.ylabel("PCB concentration in the lake trout (ppm)")
        plt.yscale("log")
        plt.title(self.title)
        plt.legend()
        plt.grid(True, which="both")
        plt.tight_layout()
        if self.plot_filename:
            plt.savefig(self.plot_filename)
        plt.show()
