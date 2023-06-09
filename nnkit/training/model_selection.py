from __future__ import annotations

from nnkit.datasets.utils import DataLabelSet

import numpy as np


class KFold:

    def __init__(self, k: int = 5, shuffle: bool = False):
        self.k = k
        self.shuffle = shuffle

    class KFoldGenerator:

        def __init__(
            self, k: int,
            points_folds: list[np.ndarray],
            labels_folds: list[np.ndarray],
            train_batch_size: int
        ):
            self.points_folds = points_folds
            self.labels_folds = labels_folds
            self.k = k
            self.train_batch_size = train_batch_size
            self.__index = self.k-1

        def __iter__(self) -> self.KFoldGenerator:
            return self

        def __next__(self) -> tuple[DataLabelSet, DataLabelSet]:
            if self.__index < 0:
                raise StopIteration
            
            test_points = self.points_folds[self.__index]
            test_labels = self.labels_folds[self.__index]

            left_points_folds = self.points_folds[:self.__index]
            right_points_folds = self.points_folds[self.__index+1:]
            left_labels_folds = self.labels_folds[:self.__index]
            right_labels_folds = self.labels_folds[self.__index+1:]

            train_points = np.concatenate(left_points_folds + right_points_folds)
            train_labels = np.concatenate(left_labels_folds + right_labels_folds)

            self.__index -= 1
            train_set = DataLabelSet(train_points, train_labels, name='training', batch_size=self.train_batch_size)
            test_set = DataLabelSet(test_points, test_labels, name='test', batch_size=len(test_points))
            return train_set, test_set
    
    def __call__(self, dataset: DataLabelSet) -> KFoldGenerator:
        points_folds, labels_folds = dataset.fair_divide(self.k)
        generator = self.KFoldGenerator(self.k, points_folds, labels_folds, dataset.batch_size)
        return generator



        

            
