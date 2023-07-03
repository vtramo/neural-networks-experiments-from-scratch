from __future__ import annotations

import numpy as np

from nnkit.datasets.utils import DataLabelSet
from nnkit.neuronet import DenseNetwork
from nnkit.training.neurotrain import NetworkTrainer

class KFold():
    def __init__(self, k: int = 5, shuffle: bool = False):
        self.k = k
        self.shuffle = shuffle

    def __get_fold_indices(self, num_samples: int) -> list:
        indices = np.arange(num_samples)

        fold_size = num_samples // self.k
        rest = num_samples % self.k

        fold_indices = []
        start = 0

        for i in range(self.k):
            end = start + fold_size

            if i < rest:
                end += 1

            fold_indices.append(indices[start:end])
            start = end

        return fold_indices
    
    def __call__(self, dataset: DataLabelSet) -> list[tuple[DataLabelSet, DataLabelSet]]:
        fold_indices = self.__get_fold_indices(len(dataset))

        fold_datasets = []

        for i in range(self.k):
            test_indices = fold_indices[i]
            train_indices = np.concatenate(fold_indices[:i] + fold_indices[i+1:])

            train_data = dataset[train_indices]
            test_data = dataset[test_indices]

            train_data.name = f"training"
            test_data.name = f"test"

            fold_datasets.append((train_data, test_data))

        return fold_datasets


        

            
