from .neuronet import DenseNetwork
from .lossfun import LossFunction
from .neurotrain import DataLabelSet
import numpy as np


def one_hot(labels, tot_classes=10):
    base = np.arange(tot_classes)
    return np.array([(base == label) * 1 for label in labels])


def fair_divide(*iterables: list, workers: int) -> list[list]:
    if len(iterables) == 0:
        return []

    def fair_partition(iterable) -> list[list]:
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

    return tuple(map(fair_partition, iterables))
