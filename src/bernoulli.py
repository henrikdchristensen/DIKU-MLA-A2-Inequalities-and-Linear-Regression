import numpy as np
import matplotlib.pyplot as plt
from math import comb


class BernoulliExperiment:
    def __init__(self, start, stop, step, bias):
        self.start = start
        self.stop = stop
        self.step = step
        self.bias = bias
        self.alphas = np.arange(self.start, self.stop + self.step, self.step)
        self.empirical_frequencies = []
        self.markov_bounds = []
        self.chebyshev_bounds = []
        self.hoeffding_bounds = []
        self.num_trials = 20

    def run_experiment(self):
        # Number of repetitions
        num_repetitions = 1000000
        # Number of Bernoulli trials in each experiment

        for alpha in self.alphas:
            experiment_results = np.random.binomial(1, self.bias, size=(num_repetitions, self.num_trials))

            # Calculate mean
            mean_results = np.mean(experiment_results, axis=1)

            # Calculate empirical frequency
            count = np.sum(mean_results >= alpha)
            empirical_frequency = count / num_repetitions
            self.empirical_frequencies.append(empirical_frequency)

            # Calculate Markov's bound
            # P(X >= alpha) <= E[X] / alpha
            markov_bound = np.mean(mean_results) / alpha
            self.markov_bounds.append(min(markov_bound, 1))  # ensure the bound is <= 1

            # Calculate Chebyshev's bound
            # P(|X - E[X]| >= k) <= Var(X) / k^2
            # Where k is the distance from the mean (i.e. standard deviation)
            # k = |alpha - E[X]|
            var_X = np.var(mean_results)
            k = np.abs(alpha - np.mean(mean_results))
            cheby_bound = var_X / k**2
            self.chebyshev_bounds.append(min(cheby_bound, 1))  # ensure the bound is <= 1

            # Calculate Hoeffding's bound
            # P(|X - E[X]| >= k) <= 2 * exp(-2 * k^2 * n)
            # Where k is the distance from the mean (i.e. standard deviation)
            # k = |alpha - E[X]|
            k = np.abs(alpha - np.mean(mean_results))
            hoeffding_bound = 2 * np.exp(-2 * k**2 * self.num_trials)
            self.hoeffding_bounds.append(min(hoeffding_bound, 1))  # ensure the bound is <= 1

    def exact_probability(self, alpha):
        threshold = int(self.num_trials * alpha)
        prob = sum(
            comb(self.num_trials, k) * self.bias**k * (1 - self.bias) ** (self.num_trials - k) for k in range(threshold, self.num_trials + 1)
        )
        return prob

    def report(self):
        for alpha, emp_freq, markov, cheby, hoeff in zip(
            self.alphas, self.empirical_frequencies, self.markov_bounds, self.chebyshev_bounds, self.hoeffding_bounds
        ):
            print(f"Alpha: {alpha}")
            print(f"Empirical Frequency: {emp_freq}")
            print(f"Markov's Bound: {markov}")
            print(f"Chebyshev's Bound: {cheby}")
            print(f"Hoeffding's Bound: {hoeff}")
            print("-" * 50)

        print(f"Exact Probability: {self.exact_probability(0.95)}")
        print(f"Exact Probability: {self.exact_probability(1)}")

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.alphas, self.empirical_frequencies, marker="o", linestyle="-", label="Empirical Frequency")
        plt.plot(self.alphas, self.markov_bounds, marker="x", linestyle="--", label="Markov's Bound")
        plt.plot(self.alphas, self.chebyshev_bounds, marker="s", linestyle="-.", label="Chebyshev's Bound")
        plt.plot(self.alphas, self.hoeffding_bounds, marker="^", linestyle=":", label="Hoeffding's Bound")
        plt.xlabel("Alpha")
        plt.ylabel("Frequency / Bound")
        plt.title("Empirical Frequency vs. Bounds")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
