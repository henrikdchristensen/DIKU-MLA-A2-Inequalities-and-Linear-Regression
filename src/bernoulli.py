import numpy as np
import matplotlib.pyplot as plt
from math import comb


class BernoulliExperiment:
    def __init__(self, start, stop, step, bias, exact_probability_for_alpha_value, plot_filename=None):
        # Filename of the plot to be saved (if None, no plot is saved)
        self.plot_filename = plot_filename

        # Alpha values to be tested
        self.start = start
        self.stop = stop
        self.step = step
        self.bias = bias
        self.alphas = np.arange(self.start, self.stop + self.step, self.step)

        # Empirical frequency and bounds for each alpha value
        self.empirical_frequencies = []
        self.markov_bounds = []
        self.chebyshev_bounds = []
        self.hoeffding_bounds = []

        # Calculate exact probabilities for the following alpha values
        self.exact_probability_for_alpha_value = exact_probability_for_alpha_value

        # Number of i.i.d. Bernoulli random variables being drawn in each repetition of the experiment
        self.n = 20

        # Number of repetitions the entire experiment (i.e. number of drawing self.n Bernoulli random variables) is repeated.
        self.num_repetitions = 1000000

        # Title for the experiment
        self.title = "Bernoulli Experiment"
        self.subtitle = (
            f"α ∈ {{{self.start:.2f}, {self.start + self.step:.2f}, ...,"
            f" {self.stop:.2f}}}, bias={self.bias:.2f}"
        )

    def run_experiment(self):
        for alpha in self.alphas:
            # Simulate Bernoulli trials
            # generate a matrix of size (num_repetitions, n) where each element is either 0 or 1
            # each row represents the result of a single experiment
            # each column represents the result of a single trial
            experiment_results = np.random.binomial(1, self.bias, size=(self.num_repetitions, self.n))

            # Mean of each experiment
            # X = [X[0], X[1], ..., X[num_repetitions - 1]]
            # X[i] = mean of the i-th experiment
            X = np.mean(experiment_results, axis=1)  # axis=1 -> mean of each row

            # Expectation of random variable
            # E[X] = 1/n * sum(X[i])
            EX = np.mean(X)

            # Empirical frequency
            count = np.sum(X >= alpha)
            empirical_frequency = count / self.num_repetitions
            self.empirical_frequencies.append(empirical_frequency)

            # Markov's bound: P(X >= α) <= E[X] / α
            markov_bound = EX / alpha
            self.markov_bounds.append(min(markov_bound, 1))

            # Chebyshev's bound: P(|X - E[X]| >= k) <= Var(X) / k^2
            # k = |alpha - E[X]|
            var_X = np.var(X)
            k = np.abs(alpha - EX)
            cheby_bound = var_X / k**2
            self.chebyshev_bounds.append(min(cheby_bound, 1))

            # Hoeffding's bound: P(|X - E[X]| >= k) <= 2 * exp(-2 * k^2 * n)
            # Using corollabry 2.5
            hoeffding_bound = 2 * np.exp(-2 * k**2 * self.n)
            self.hoeffding_bounds.append(min(hoeffding_bound, 1))

    def exact_probability(self, alpha):
        # P(X >= α) = sum(comb(n, k) * p^k * (1 - p)^(n - k))
        return sum(
            comb(self.n, k) * self.bias**k * (1 - self.bias) ** (self.n - k)
            for k in range(int(self.n * alpha), self.n + 1)
        )

    def report(self):
        print("-" * 50)
        print(self.title)
        for alpha in self.exact_probability_for_alpha_value:
            print(f"P(X >= {alpha:.2f}) ≈ {self.exact_probability(alpha):.2e}")
        print("-" * 50)

    def plot(self):
        # Size of the plot
        plt.figure(figsize=(10, 6))

        # Plot empirical frequency and bounds
        plt.plot(
            self.alphas, self.empirical_frequencies, marker="o", linestyle="-", label="Empirical frequency"
        )
        plt.plot(self.alphas, self.markov_bounds, marker="x", linestyle="--", label="Markov's bound")
        plt.plot(self.alphas, self.chebyshev_bounds, marker="s", linestyle="-.", label="Chebyshev's bound")
        plt.plot(self.alphas, self.hoeffding_bounds, marker="^", linestyle=":", label="Hoeffding's bound")

        # Plot settings
        plt.xlabel("α", fontsize=12)
        plt.xticks(self.alphas)
        plt.ylabel("Frequency (prob.)", fontsize=12)
        plt.suptitle(self.title, fontsize=14, x=0.53)
        plt.title(self.subtitle, fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if self.plot_filename:
            plt.savefig(self.plot_filename)
        plt.show()
