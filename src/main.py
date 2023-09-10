from bernoulli import BernoulliExperiment
from linear_regression import LinearRegression
import numpy as np


def main():
    # First experiment
    experiment1 = BernoulliExperiment(start=0.5, stop=1, step=0.05, bias=0.5)
    experiment1.run_experiment()
    experiment1.report()
    experiment1.plot()

    # Second experiment
    experiment2 = BernoulliExperiment(start=0.1, stop=1, step=0.05, bias=0.1)
    experiment2.run_experiment()
    experiment2.report()
    experiment2.plot()

    # Load the data from PCB.dt - each line contains a pair (x, y)
    data = np.loadtxt("data/PCB.dt")

    # Linear model
    linear_model = LinearRegression(data)
    linear_model.fit_model()
    linear_model.report()
    linear_model.plot()

    # Non-linear model: x -> sqrt(x)
    non_linear_model = LinearRegression(data, X_map_func=np.sqrt)
    non_linear_model.fit_model()
    non_linear_model.report()
    non_linear_model.plot()


if __name__ == "__main__":
    main()
