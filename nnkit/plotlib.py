from nnkit.training.neurotrain import TrainingHistory
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    train_history: TrainingHistory,
    metrics: set[str] = None,
    show_plot: bool = True,
    path: str = ''
):

    def must_be_plotted(metric_name: str) -> bool:
        return metrics is None or metric_name in metrics

    metric_values_by_name = {}
    for metric_by_name in train_history.history:
        for metric_name, metric in metric_by_name.items():
            if not must_be_plotted(metric_name):
                continue
            metric_values_by_name.setdefault(metric_name, [])
            metric_values = metric_values_by_name[metric_name]
            metric_values.append(metric.result())

    epochs = np.arange(1, train_history.epochs + 1)
    for metric_name, metric_values in metric_values_by_name.items():
        plt.plot(epochs, metric_values, '-', label=metric_name, lw=1.2)

    plt.xticks(epochs)
    plt.xlabel('Epochs', color='black')
    plt.ylabel('Metrics', color='black')
    plt.title(train_history.update_rule)
    plt.legend()

    if show_plot:
        plt.show()

    if path:
        plt.savefig(path)

