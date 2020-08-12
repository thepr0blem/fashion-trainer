from typing import List

import matplotlib.pyplot as plt


def plot_loss(iterations: List, losses: List):
    plot_metric(iterations=iterations, metric=losses, metric_name="Loss")


def plot_accuracy(iterations: List, accuracies: List):
    plot_metric(iterations=iterations, metric=accuracies, metric_name="Accuracy [%]")


def plot_metric(iterations: List, metric: List, metric_name: str):

    plt.plot(iterations, metric)
    plt.xlabel("No. of Iteration")
    plt.ylabel(metric_name)
    plt.title(f"Iterations vs {metric_name}")
    plt.savefig(f"figures/{metric_name.lower()}.png")
    plt.clf()
