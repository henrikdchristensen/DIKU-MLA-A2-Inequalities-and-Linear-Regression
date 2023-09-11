from bernoulli import BernoulliExperiment
from linear_regression import LinearRegression
import numpy as np


def main():
    # Bernoulli experiment: bias = 0.5
    experiment1 = BernoulliExperiment(start=0.5, stop=1, step=0.05, bias=0.5, plot_filename="output/bernoulli/bias_0.5.png")
    experiment1.run_experiment()
    experiment1.report()
    experiment1.plot()

    # Bernoulli experiment: bias = 0.1
    experiment2 = BernoulliExperiment(start=0.1, stop=1, step=0.05, bias=0.1, plot_filename="output/bernoulli/bias_0.1.png")
    experiment2.run_experiment()
    experiment2.report()
    experiment2.plot()

    # Load the data from PCB.dt - each line contains a pair (x, y)
    data = np.loadtxt("data/PCB.dt")

    # Linear model
    linear_model = LinearRegression(data, plot_filename="output/linear_regression/x_to_x.png")
    linear_model.fit_model()
    linear_model.report()
    linear_model.plot()

    # Non-linear model: x -> sqrt(x)
    non_linear_model = LinearRegression(data, X_map_func=np.sqrt, plot_filename="output/linear_regression/x_to_sqrt_x.png")
    non_linear_model.fit_model()
    non_linear_model.report()
    non_linear_model.plot()

    # Non-linear model: x -> log(x)
    non_linear_model2 = LinearRegression(data, X_map_func=np.log, plot_filename="output/linear_regression/x_to_log_x.png")
    non_linear_model2.fit_model()
    non_linear_model2.report()
    non_linear_model2.plot()


if __name__ == "__main__":
    main()
