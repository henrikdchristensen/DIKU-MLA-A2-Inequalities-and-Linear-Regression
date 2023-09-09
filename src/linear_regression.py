import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, X, y_true, y_model_type="linear", X_model_type="linear"):
        self.weights = None
        self.y_model_type = y_model_type
        self.y_pred = None
        self.y_true = y_true
        self.X = X
        self.y_transformed = None
        self.X_transformed = None
        self.X_model_type = X_model_type

    def _X_transform(self, X):
        if self.X_model_type == "sqrt":
            self.X_transformed = np.sqrt(X)
        elif self.X_model_type == "linear":
            self.X_transformed = X

    def fit(self):
        # Transform the input X and output y
        self._X_transform(self.X)
        y_to_fit = np.log(self.y_true) if self.y_model_type == "log" else self.y_true

        # Add a column of ones for the bias/intercept term.
        X_b = np.c_[np.ones((self.X_transformed.shape[0], 1)), self.X_transformed]

        # Compute the weights using the normal equation
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_to_fit)

        # Add a column of ones to X to account for the bias/intercept term.
        X_b = np.c_[np.ones((self.X_transformed.shape[0], 1)), self.X_transformed]

        # Compute the weights using the normal equation
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_to_fit)

    def predict(self):
        # Predict values using the learned weights
        X_b = np.c_[np.ones((self.X_transformed.shape[0], 1)), self.X_transformed]
        self.y_pred = X_b.dot(self.weights)

        self.y_pred = np.exp(self.y_pred) if self.y_model_type == "log" else self.y_pred

    def _mean_squared_error(self):
        return np.mean((self.y_true - self.y_pred) ** 2)

    def _r_squared(self):
        residual_sum_of_squares = np.sum((self.y_true - self.y_pred) ** 2)
        total_sum_of_squares = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
        return r2

    def report(self):
        print(f"Parameters (weights): {self.weights}")
        print(f"Mean Squared Error: {self._mean_squared_error()}")
        print(f"R^2: {self._r_squared()}")

    def plot(self):
        # Sort the data by X values for plotting
        sorted_indices = np.argsort(self.X[:, 0])
        X_sorted = self.X[sorted_indices]
        y_sorted = self.y_true[sorted_indices]
        y_pred_sorted = self.y_pred[sorted_indices]

        # Plotting original data points
        plt.scatter(X_sorted, y_sorted, color="blue", label="Original data")
        # Plotting the linear regression line
        plt.plot(X_sorted, y_pred_sorted, color="red", label="Fitted line", linewidth=2)
        plt.xlabel("Age of fish")
        plt.ylabel("PCB Concentration (in ppm)")
        plt.title("Linear Regression of PCB Concentration vs. Age of Fish")
        plt.legend()
        plt.show()

    def plot_log(self):
        # Log-transform the y-values
        y_log = np.log(self.y_true)

        # Plot the actual data points
        plt.scatter(self.X, y_log, color="blue", label="Actual data")

        # Get the model's predictions (make sure you've called `predict` method before plotting)
        y_pred_log = np.log(self.y_pred)

        # Plot the model's predictions
        plt.plot(self.X, y_pred_log, color="red", label="Model output")

        # Setting labels, title, and legend
        plt.xlabel("Age of fish (not transformed)")
        plt.ylabel("Log of PCB Concentration")
        plt.title("Log of PCB Concentration vs. Age of Fish")
        plt.legend()

        # Display the plot
        plt.show()

    def plot_sqrt_log(self):
        y_log = np.log(self.y_true)

        # Sort the data by X values for plotting
        sorted_indices = np.argsort(self.X_transformed[:, 0])
        X_sorted = self.X_transformed[sorted_indices]
        y_sorted = y_log[sorted_indices]
        y_pred_sorted = y_log[sorted_indices]

        # Plotting original data points
        plt.scatter(X_sorted, y_sorted, color="blue", label="Original data")
        # Plotting the linear regression line
        plt.plot(X_sorted, y_pred_sorted, color="red", label="Fitted line", linewidth=2)
        plt.xlabel("Sqrt(Age of fish)")
        plt.ylabel("Log of PCB Concentration")
        plt.title("Log of PCB Concentration vs. Sqrt(Age of Fish)")
        plt.legend()
        plt.show()
