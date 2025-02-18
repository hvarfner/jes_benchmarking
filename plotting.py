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

def plot_results(benchmarks, acquisition_functions, metric, base_dirs):
    fig, axes = plt.subplots(len(benchmarks), 1, figsize=(10, 6 * len(benchmarks)))

    if len(benchmarks) == 1:
        axes = [axes]

    markers = ['o', '']
    linestyles = ['-', '--']

    for ax, benchmark in zip(axes, benchmarks):
        for idx, base_dir in enumerate(base_dirs):
            for acq_func in acquisition_functions:
                directory = os.path.join(base_dir, benchmark, acq_func)
                results = load_results(directory)
                iterations, averages, std = compute_average(results, metric)
                ax.plot(
                    iterations,
                    averages,
                    label=f"{acq_func} ({base_dir})",
                    color=colors[acq_func],
                    marker=markers[idx],
                    linestyle=linestyles[idx]
                )
                ax.fill_between(
                    iterations,
                    averages - std,
                    averages + std,
                    alpha=0.2,
                    color=colors[acq_func],
                )

        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} for {benchmark}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    benchmarks = ["Levy_4", "Hartmann_6"]
    benchmarks = ["Ackley_8", "Ackley_16", "Michalewicz_5", "Michalewicz_10"]
    acquisition_functions = ["jes", "logei"]
    metric = "best_f"  # Choose from 'best_f', 'out_of_sample_f', 'in_sample_f'
    base_dirs = ["results_noiseless",  "old_results"]
    plot_results(benchmarks, acquisition_functions, metric, base_dirs)
