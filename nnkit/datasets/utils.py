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

    def __init__(self, points: np.ndarray, labels: np.ndarray, name: str = ""):
        assert len(points) == len(labels)
        self._points = points
        self._labels = labels
        self.name = name

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        return self._points, self._labels

    def fair_divide(self, workers: int) -> tuple[list[list], list[list]]:
        return fair_divide(self._points, self._labels, workers=workers)

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> zip[np.ndarray, np.ndarray]:
        return zip(self._points, self._labels)
    
    def split(self, split_factor: float, split_set_name: str = "") -> tuple[DataLabelSet, DataLabelSet]:
        split_index = int(len(self) * split_factor)
        left_dataset = DataLabelSet(self._points[split_index:], self._labels[split_index:], name=self.name)
        right_dataset = DataLabelSet(self._points[:split_index], self._labels[:split_index], name=split_set_name)
        return left_dataset, right_dataset


class DataLabelBatchGenerator(DataLabelSet):

    def __init__(self, points: np.ndarray, labels: np.ndarray, batch_size=128, name: str = ""):
        super().__init__(points, labels, name)
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
    
    @classmethod
    def from_data_label_set(cls, data_label_set: DataLabelSet, batch_size=128):
        return cls(data_label_set._points, data_label_set._labels, batch_size=batch_size, name=data_label_set.name)

    def split(self, split_factor: float, split_set_name="") -> tuple[DataLabelBatchGenerator, DataLabelBatchGenerator]:
        left_dataset, right_dataset = super().split(split_factor)
        left_dataset_points, left_dataset_labels = left_dataset.get()
        right_dataset_points, right_dataset_labels = right_dataset.get()
        left_dataset_batch_generator = DataLabelBatchGenerator(left_dataset_points, left_dataset_labels, batch_size=self._batch_size, name=self.name)
        right_dataset_batch_generator = DataLabelBatchGenerator(left_dataset_points, left_dataset_labels, batch_size=self._batch_size, name=split_set_name)
        return left_dataset_batch_generator, right_dataset_batch_generator
    
