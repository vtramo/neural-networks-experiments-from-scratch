import numpy as np
import math


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


class DataLabelSet:

    def __init__(self, points, labels):
        assert len(points) == len(labels)
        self._points = points
        self._labels = labels

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        return self._points, self._labels

    def fair_divide(self, workers: int) -> tuple[list[list], list[list]]:
        return fair_divide(self._points, self._labels, workers=workers)

    def __len__(self):
        return len(self._points)


class DataLabelBatchGenerator(DataLabelSet):

    def __init__(self, points, labels, batch_size=128):
        super().__init__(points, labels)
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(points) / batch_size)

    class DataLabelIterator:

        def __init__(self, outer_instance):
            self.index = 0
            self.outer_instance = outer_instance

        def __next__(self):
            points = self.outer_instance._points[self.index: self.index + self.outer_instance.batch_size]
            labels = self.outer_instance._labels[self.index: self.index + self.outer_instance.batch_size]

            if len(points) == 0:
                raise StopIteration

            self.index += self.outer_instance.batch_size
            return points, labels

    def __iter__(self) -> DataLabelIterator:
        return self.DataLabelIterator(self)
