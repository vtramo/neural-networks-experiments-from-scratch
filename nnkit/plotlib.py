from nnkit.training.neurotrain import TrainingHistory

import matplotlib.pyplot as plt
import numpy as np
import pickle


def load_histories_from_file(path: str) -> list[TrainingHistory]:
    with open(path, 'rb') as file:
        histories = pickle.load(file)
    return histories


def save_histories_to_file(histories: list[TrainingHistory], path: str):
    with open(path, 'wb') as file:
        pickle.dump(histories, file)


def plot_training_history(
    train_history: TrainingHistory,
    metrics: set[str] = None,
    show_plot: bool = True,
    path: str = '',
    title: str = '',
    xlabel: str = '',
    ylabel: str = ''
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

    ax = plt.axes()
    ax.set_xlim(0, max(epochs))
    ax.set_ylim(0, 1)
    ax.grid(linestyle='--')

    for metric_name, metric_values in metric_values_by_name.items():
        plt.plot(epochs, metric_values, '-', label=metric_name, lw=1.2)

    plt.xticks(np.arange(min(epochs)-1, max(epochs)+1, 10))
    plt.xlabel('Epochs' if not xlabel else xlabel, color='black')
    plt.ylabel('Metrics' if not ylabel else ylabel, color='black')
    plt.title(train_history.update_rule if not title else title)
    plt.legend()

    if show_plot:
        plt.show()

    if path:
        plt.savefig(path)


def plot_training_histories(
    train_histories: list[TrainingHistory],
    metric: str,
    show_plot: bool = True,
    path: str = '',
    title: str = '',
    xlabel: str = '',
    ylabel: str = ''
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

    ax = plt.axes()
    ax.set_xlim(0, max(epochs))
    ax.set_ylim(0, 1)
    ax.grid(linestyle='--')

    for metric_name, metric_values in metric_values_by_name.items():
        plt.plot(epochs, metric_values, '-', label=metric_name, lw=1.2)

    plt.xticks(np.arange(min(epochs)-1, max(epochs)+1, 10))
    plt.xlabel('Epochs' if not xlabel else xlabel, color='black')
    plt.ylabel(metric if not ylabel else ylabel, color='black')
    plt.title(f'{metric} across Training' if not title else title)
    plt.legend()

    if show_plot:
        plt.show()

    if path:
        plt.savefig(path)
