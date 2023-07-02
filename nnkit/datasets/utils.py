from __future__ import annotations

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

    def __init__(self, points: np.ndarray, labels: np.ndarray, batch_size: int = 128, name: str = ""):
        assert len(points) == len(labels)
        self._points = points
        self._labels = labels
        self.name = name
        self.batch_size = batch_size
        self.steps = math.ceil(len(points) / batch_size)

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        return self._points, self._labels

    def fair_divide(self, workers: int) -> tuple[list[list], list[list]]:
        return fair_divide(self._points, self._labels, workers=workers)

    def __len__(self) -> int:
        return len(self._points)

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

    def __iter__(self) -> zip[np.ndarray, np.ndarray]:
        return self.DataLabelIterator(self)
    
    def split(
        self,
        split_factor: float,
        split_set_batch_size: int = None,
        split_set_name: str = ""
    ) -> tuple[DataLabelSet, DataLabelSet]:
        split_index = int(len(self) * split_factor)

        left_dataset = DataLabelSet(
            self._points[split_index:],
            self._labels[split_index:],
            name=self.name,
            batch_size=self.batch_size
        )

        split_set_batch_size = split_set_batch_size if split_set_batch_size is not None else self.batch_size
        right_dataset = DataLabelSet(
            self._points[:split_index],
            self._labels[:split_index],
            name=split_set_name,
            batch_size=split_set_batch_size
        )

        return left_dataset, right_dataset
