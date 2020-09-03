from typing import List

import matplotlib.pyplot as plt


def save_loss_plot(iterations: List, losses: List) -> None:
    plt.plot(iterations, losses)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Loss")
    plt.title("Iterations vs Loss")
    plt.savefig("figures/loss.png")
    plt.clf()


def save_accuracy_plot(
    train_accuracy: List, test_accuracy: List, iterations: List,
) -> None:
    plt.plot(iterations, train_accuracy)
    plt.plot(iterations, test_accuracy)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Accuracy [%]")
    plt.legend(["train", "val"], loc="upper left")
    plt.title("Accuracy")
    plt.savefig("figures/accuracy.png")
    plt.clf()
