import json
import os

import matplotlib.pyplot as plt
import numpy as np

colors = {
    "jes": "red",
    "logei": "orange",
    "pes": "green",
}


def load_results(directory):
    results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    results.append(data)
    return results


def compute_average(results, metric, num_err=2):
    iteration_data = {}
    for run in results:
        for entry in run:
            iteration = entry["iteration"]
            value = entry.get(metric, None)
            if value is not None:
                iteration_data.setdefault(iteration, []).append(value)

    iterations = sorted(iteration_data.keys())
    averages = np.array([np.mean(iteration_data[i]) for i in iterations])
    std = num_err * np.array(
        [np.std(iteration_data[i]) / len(iteration_data[i]) for i in iterations]
    )
    return iterations, averages, std


def plot_results(
    benchmark, acquisition_functions, metric, base_dir="results_noiseless"
):
    plt.figure(figsize=(10, 6))
    for acq_func in acquisition_functions:
        directory = os.path.join(base_dir, benchmark, acq_func)
        results = load_results(directory)
        iterations, averages, std = compute_average(results, metric)
        plt.plot(iterations, averages, label=f"{acq_func}", color=colors[acq_func])
        plt.fill_between(
            iterations,
            averages - std,
            averages + std,
            alpha=0.2,
            color=colors[acq_func],
        )
        plt.plot(iterations, averages - std, alpha=0.3, color=colors[acq_func])
        plt.plot(iterations, averages + std, alpha=0.3, color=colors[acq_func])

    plt.xlabel("Iteration")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} for {benchmark}")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    benchmark = "Hartmann_4"
    acquisition_functions = ["jes", "pes", "logei"]
    metric = "out_of_sample_f"  # Choose from 'best_f', 'out_of_sample_f', 'in_sample_f'
    plot_results(benchmark, acquisition_functions, metric)
