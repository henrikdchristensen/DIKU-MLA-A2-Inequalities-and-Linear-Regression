import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, nonlinear):
        self.weights = None
        self.nonlinear = nonlinear
        self.y_pred = None

    def fit(self, X, y):
        if self.nonlinear:
            # Transform the y-values using the natural logarithm.
            y_to_fit = np.log(y)
        else:
            y_to_fit = y

        # Add a column of ones to X to account for the bias/intercept term.
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Compute the weights using the normal equation
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_to_fit)

    def predict(self, X):
        # Predict values using the learned weights
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = X_b.dot(self.weights)

        if self.nonlinear:
            self.y_pred = np.exp(y_pred)
        else:
            self.y_pred = y_pred

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def report(self, y_true):
        mse = self.mean_squared_error(y_true, self.y_pred)
        print(f"Parameters (weights): {self.weights}")
        print(f"Mean Squared Error: {mse}")

    def plot(self, X, y):
        # Sort the data by X values for plotting
        sorted_indices = np.argsort(X[:, 0])
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
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

    def plot_log(self, X, y):
        # Log-transform the y-values
        y_log = np.log(y)

        # Plot the actual data points
        plt.scatter(X, y_log, color="blue", label="Actual data")

        # Get the model's predictions (make sure you've called `predict` method before plotting)
        y_pred_log = np.log(self.y_pred)

        # Plot the model's predictions
        plt.plot(X, y_pred_log, color="red", label="Model output")

        # Setting labels, title, and legend
        plt.xlabel("Age of fish (not transformed)")
        plt.ylabel("Log of PCB Concentration")
        plt.title("Log of PCB Concentration vs. Age of Fish")
        plt.legend()

        # Display the plot
        plt.show()
