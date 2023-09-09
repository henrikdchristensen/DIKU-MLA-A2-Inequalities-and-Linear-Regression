import numpy as np
import matplotlib.pyplot as plt


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

    def run_experiment(self):
        # Number of repetitions
        num_repetitions = 1000000
        # Number of Bernoulli trials in each experiment
        num_trials = 20

        for alpha in self.alphas:
            experiment_results = np.random.binomial(1, self.bias, size=(num_repetitions, num_trials))

            # Calculate mean
            mean_results = np.mean(experiment_results, axis=1)

            # Calculate empirical frequency
            count = np.sum(mean_results >= alpha)
            empirical_frequency = count / num_repetitions
            self.empirical_frequencies.append(empirical_frequency)

            # Calculate Markov's bound
            expected_value = np.mean(mean_results)
            markov_bound = expected_value / alpha
            self.markov_bounds.append(min(markov_bound, 1))  # ensure the bound is <= 1

            # Calculate Chebyshev's bound
            k = np.abs(alpha - np.mean(mean_results)) / np.sqrt(np.var(mean_results))
            chebyshev_bound = 1 / (k**2)
            self.chebyshev_bounds.append(min(chebyshev_bound, 1))  # ensure the bound is <= 1

            # Calculate Hoeffding's bound
            hoeffding_bound = np.exp(-2 * num_trials * (alpha - 0.5) ** 2)
            self.hoeffding_bounds.append(hoeffding_bound)

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

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.alphas, self.empirical_frequencies, marker="o", linestyle="-", label="Empirical Frequency")
        plt.plot(self.alphas, self.markov_bounds, marker="x", linestyle="--", label="Markov's Bound")
        plt.plot(self.alphas, self.chebyshev_bounds, marker="s", linestyle="-.", label="Chebyshev's Bound")
        plt.plot(self.alphas, self.hoeffding_bounds, marker="^", linestyle=":", label="Hoeffding's Bound")
        plt.xlabel("Alpha")
        plt.ylabel("Frequency / Bound")
        plt.title("Empirical Frequency vs. Bounds (Bounded at 1)")
        plt.legend()
        plt.grid(True)
        plt.show()
