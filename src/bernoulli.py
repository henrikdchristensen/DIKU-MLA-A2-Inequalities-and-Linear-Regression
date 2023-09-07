import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt


def bernoulli_experiments(start, stop, step, bias):
    # Number of repetitions
    num_repetitions = 1000000

    # Number of Bernoulli trials in each experiment
    num_trials = 20

    # Bias for the Bernoulli trials
    bias = 0.5  # 1/2

    # List of alpha values to investigate
    alphas = np.arange(start, stop + step, step)

    # Initialize lists to store the empirical frequencies, Markov's bounds, Chebyshev's bounds, and Hoeffding's bounds
    empirical_frequencies = []
    markov_bounds = []
    chebyshev_bounds = []
    hoeffding_bounds = []

    # Perform the experiment 1,000,000 times
    for alpha in alphas:
        # Simulate 20 Bernoulli trials using scipy.stats.bernoulli
        experiment_results = bernoulli.rvs(bias, size=(num_repetitions, num_trials))

        # Calculate the mean of each experiment
        mean_results = np.mean(experiment_results, axis=1)

        # Calculate the empirical frequency
        count = np.sum(mean_results >= alpha)
        empirical_frequency = count / num_repetitions
        empirical_frequencies.append(empirical_frequency)

        # Calculate Markov's bound
        expected_value = np.mean(mean_results)
        markov_bound = expected_value / alpha
        markov_bounds.append(min(markov_bound, 1))  # Ensure the bound is <= 1

        # Calculate Chebyshev's bound
        k = np.abs(alpha - np.mean(mean_results)) / np.sqrt(np.var(mean_results))
        chebyshev_bound = 1 / (k**2)
        chebyshev_bounds.append(min(chebyshev_bound, 1))  # Ensure the bound is <= 1

        # Calculate Hoeffding's bound
        hoeffding_bound = np.exp(-2 * num_trials * (alpha - 0.5) ** 2)
        hoeffding_bounds.append(hoeffding_bound)

    # Plot the empirical frequencies and bounds (with bounds capped at 1)
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, empirical_frequencies, marker="o", linestyle="-", label="Empirical Frequency")
    plt.plot(alphas, markov_bounds, marker="x", linestyle="--", label="Markov's Bound")
    plt.plot(alphas, chebyshev_bounds, marker="s", linestyle="-.", label="Chebyshev's Bound")
    plt.plot(alphas, hoeffding_bounds, marker="^", linestyle=":", label="Hoeffding's Bound")
    plt.xlabel("Alpha")
    plt.ylabel("Frequency / Bound")
    plt.title("Empirical Frequency vs. Bounds (Bounded at 1)")
    plt.legend()
    plt.grid(True)
    plt.show()


start = 0.5
stop = 1.0
step = 0.05
bias = 0.5
bernoulli_experiments(start, stop, step, bias)

start = 0.1
stop = 1.0
step = 0.05
bias = 0.1
bernoulli_experiments(start, stop, step, bias)


# Task 2
# The granularity of alpha values in the code (e.g., 0.5, 0.55, 0.6, ...)
# is typically sufficient because it allows for meaningful insights without
# adding unnecessary computational complexity. Choosing smaller increments (e.g., 0.51)
# is unlikely to yield substantially different results unless there are
# specific reasons to investigate at such fine levels, and it may increase
# computational demands.
