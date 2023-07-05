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


def plot_training_histories(
    train_histories: list[TrainingHistory],
    metric: str,
    show_plot: bool = True,
    path: str = ''
):
    metric_values_by_name = {}
    epochs = np.arange(1, train_histories[0].epochs + 1)
    
    for train_history in train_histories:
        for _ in train_history.history:
            for metric_name, metrics in _.items():
                if metric_name == metric:
                    new_metric_name = f'{train_history.update_rule}_{metric_name}'
                    metric_values_by_name.setdefault(new_metric_name, []) 
                    metric_values_by_name[new_metric_name].append(metrics.result())

    for metric_name, metric_values in metric_values_by_name.items():
            print(metric_name, metric_values)
            plt.plot(epochs, metric_values, '-', label=metric_name, lw=1.2)

    plt.xticks(epochs)
    plt.xlabel('Epochs', color='black')
    plt.ylabel(metric, color='black')
    plt.title(f'{metric} across Training')
    plt.legend()

    if show_plot:
        plt.show()

    if path:
        plt.savefig(path)