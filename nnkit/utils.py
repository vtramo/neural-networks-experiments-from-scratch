from .neuronet import DenseNetwork
from .lossfun import LossFunction
import numpy as np


def one_hot(labels, tot_classes=10):
    base = np.arange(tot_classes)
    return np.array([(base == label) * 1 for label in labels])


def fair_divide(iterable, workers: int) -> list[list, ...]:
    partition = []
    count = len(iterable) // workers
    remainder = len(iterable) % workers

    for i in range(workers):
        if i < remainder:
            start = i * (count + 1)
            stop = start + count
        else:
            start = i * count + remainder
            stop = start + (count - 1)
        partition.append(iterable[start:stop + 1])

    return partition


def calculate_accuracy_and_loss(net: DenseNetwork, points, labels, loss: LossFunction) -> tuple[float, float]:
    correct_predictions = 0
    total_loss = 0

    for image, label in zip(points, labels):
        net_output = net(image)

        if np.argmax(net_output) == np.argmax(label):
            correct_predictions += 1

    return correct_predictions / len(points), total_loss / len(points)
