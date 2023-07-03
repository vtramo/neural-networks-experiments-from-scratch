from __future__ import annotations

import numpy as np
from pyparsing import Iterator

from nnkit.datasets.utils import DataLabelSet
from nnkit.neuronet import DenseNetwork
from nnkit.training.neurotrain import NetworkTrainer

class KFold():
    def __init__(self, k: int = 5, shuffle: bool = False):
        self.k = k
        self.shuffle = shuffle

    class KFoldGenerator(Iterator[tuple[DataLabelSet, DataLabelSet]]):
        def __init__(self, k: int, points_folds: list[np.ndarray], labels_folds: list[np.ndarray]):
            self.points_folds = points_folds
            self.labels_folds = labels_folds
            self.k = k
            self.__index = self.k-1

        def __next__(self):
            if self.__index < 0:
                raise StopIteration
            
            test_points = self.points_folds[self.__index]
            test_labels = self.labels_folds[self.__index]

            left_points = [] if self.__index == -1 else self.points_folds[:self.__index]
            right_points = [] if self.__index + 1 == self.k else self.points_folds[self.__index+1:]
            left_labels = [] if self.__index == -1 else self.labels_folds[:self.__index]
            right_labels = [] if self.__index + 1 == self.k else self.labels_folds[self.__index+1:]

            train_points = np.concatenate(left_points + right_points)
            train_labels = np.concatenate(left_labels + right_labels)

            self.__index -= 1
            
            train_set = DataLabelSet(train_points, train_labels)
            test_set = DataLabelSet(test_points, test_labels)

            train_set.name = f"training"
            test_set.name = f"test"

            return train_set, test_set
    
    def __call__(self, dataset: DataLabelSet) -> list[tuple[DataLabelSet, DataLabelSet]]:
        points_folds, labels_folds = dataset.fair_divide(self.k)
        generator = self.KFoldGenerator(self.k, points_folds, labels_folds)
        return generator



        

            
