import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, data, X_map_func=None, plot_filename=None):
        # Function to map X values to X' values (i.e. X' = X_map_func(X))
        self.X_map_func = X_map_func

        # Filename of the plot to be saved (if None, no plot is saved)
        self.plot_filename = plot_filename

        # Parameters a and b to be determined
        self.a = None
        self.b = None

        # Predictions: h(x) = exp(a * x + b)
        self.y_pred = None

        # Sort dataset by X values
        sorted_indices = np.argsort(data[:, 0])
        self.X = data[:, 0][sorted_indices]
        self.y_true = data[:, 1][sorted_indices]

        # Construct dataset S'
        self.X_prime = X_map_func(self.X) if X_map_func else self.X
        self.y_prime = np.log(self.y_true)

        # Title for the model
        self.title = ("Non-Linear" if self.X_map_func else "Linear") + " Regression of Lake trouts"
        self.subtitle = "Age (years) vs. PCB concentration (ppm)\n" + (
            f"Transformation: x → {self.X_map_func.__name__}(x)" if self.X_map_func else ""
        )

    def fit_model(self):
        # Append a column of ones to X (equation 3.3 in [CI])
        X_tilde = np.c_[np.ones(self.X_prime.shape[0]), self.X_prime]

        # Calculate w* = (X^T * X)^-1 * X^T * y and obtain parameters a and b (equation 3.5 in [CI])
        w_star = np.linalg.inv(X_tilde.T.dot(X_tilde)).dot(X_tilde.T).dot(self.y_prime)
        # For one independent variable, w* contains only two parameters (b and a)
        self.b = w_star[0]
        self.a = w_star[1]

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
        print(self.subtitle)
        print(f"a: {self.a:.4f}")
        print(f"b: {self.b:.4f}")
        print(f"MSE: {self.mse():.4f}")
        print(f"R^2: {self.r2():.4f}")
        print("-" * 50)

    def plot(self):
        # Size of the plot
        plt.figure(figsize=(10, 6))

        # Plot the true values and the predictions
        # Always use "normal" X scale
        plt.scatter(self.X, self.y_true, marker="o", label="True values")
        plt.plot(self.X, self.y_pred, marker="x", linestyle="-", color="r", label="Predictions")

        # Plot settings
        plt.xlabel("Age (years)", fontsize=12)
        plt.xticks(np.arange(min(self.X), max(self.X) + 1, 1))
        plt.ylabel("PCB concentration (ppm)", fontsize=12)
        plt.yscale("log")
        plt.suptitle(self.title, fontsize=14, x=0.53)
        plt.title(self.subtitle, fontsize=12)
        plt.legend()
        plt.grid(True, which="both")
        plt.tight_layout()
        if self.plot_filename:
            plt.savefig(self.plot_filename)
        plt.show()
