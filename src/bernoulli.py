import numpy as np
import matplotlib.pyplot as plt
from math import comb


class BernoulliExperiment:
    def __init__(self, start, stop, step, bias):
        self.start = start
        self.stop = stop
        self.step = step
        self.bias = bias
        self.alphas = np.arange(self.start, self.stop + self.step, self.step)  # threshold values
        self.empirical_frequencies = []
        self.markov_bounds = []
        self.chebyshev_bounds = []
        self.hoeffding_bounds = []
        self.n = 20  # number of Bernoulli trials
        self.num_repetitions = 1000000
        self.title = f"Bernoulli Experiment\nα ∈ {{{self.start:.2f}, {self.start + self.step:.2f}, ..., {self.stop:.2f}}}, bias={self.bias:.2f}"

    def run_experiment(self):
        for alpha in self.alphas:
            # Simulate Bernoulli trials
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
            hoeffding_bound = 2 * np.exp(-2 * k**2 * self.n)
            self.hoeffding_bounds.append(min(hoeffding_bound, 1))

    def exact_probability(self, alpha):
        # P(X >= α) = sum(comb(n, k) * p^k * (1 - p)^(n - k))
        threshold = int(self.n * alpha)  # threshold for number of successes
        prob = sum(comb(self.n, k) * self.bias**k * (1 - self.bias) ** (self.n - k) for k in range(threshold, self.n + 1))
        return prob

    def report(self):
        # Print  exact probabilities for specified alphas
        alpha_values = [0.95, 1]
        print("-" * 50)
        print(self.title)
        for alpha in alpha_values:
            print(f"P(X >= {alpha:.2f}) = {self.exact_probability(alpha):.4e}")
        print("-" * 50)

    def plot(self):
        # Plot empirical frequency and bounds
        plt.figure(figsize=(10, 6))
        plt.plot(self.alphas, self.empirical_frequencies, marker="o", linestyle="-", label="Empirical frequency")
        plt.plot(self.alphas, self.markov_bounds, marker="x", linestyle="--", label="Markov's bound")
        plt.plot(self.alphas, self.chebyshev_bounds, marker="s", linestyle="-.", label="Chebyshev's bound")
        plt.plot(self.alphas, self.hoeffding_bounds, marker="^", linestyle=":", label="Hoeffding's bound")
        plt.xlabel("α")
        plt.ylabel("Frequency / Bound")
        plt.title(self.title)
        plt.xticks(self.alphas)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
