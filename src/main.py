from bernoulli import BernoulliExperiment
from linear_regression import LinearRegression
import numpy as np


def main():
    # # First experiment
    # experiment1 = BernoulliExperiment(start=0.5, stop=1, step=0.05, bias=0.5)
    # experiment1.perform_experiment()
    # experiment1.report()
    # experiment1.plot()

    # # Second experiment
    # experiment2 = BernoulliExperiment(start=0.1, stop=1, step=0.05, bias=0.5)
    # experiment2.perform_experiment()
    # experiment2.report()
    # experiment2.plot()

    # Load the data from PCB.dt where each line contains a pair (x, y)
    data_list = [line.strip().split() for line in open("data/PCB.dt").readlines()]
    data = np.array(data_list, dtype=float)

    # Split the data into input X and output y
    X = data[:, 0].reshape(-1, 1)  # Age of the fish
    y = data[:, 1]  # PCB concentration

    model1 = LinearRegression(X, y, transform_X=False, transform_y=True)
    model1.fit_model()
    model1.predict()
    model1.report()
    model1.plot()


if __name__ == "__main__":
    main()
