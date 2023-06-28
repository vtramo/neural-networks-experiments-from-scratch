from __future__ import annotations

from typing import Generic, TypeVar
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

import numpy as np

R = TypeVar('R')


class Metrics(Generic[R], metaclass=ABCMeta):

    @abstractmethod
    def update(self, predictions: np.ndarray, label: np.ndarray) -> Metrics:
        pass

    @abstractmethod
    def combine(self, metric: Metrics) -> Metrics:
        pass

    @abstractmethod
    def result(self) -> R:
        pass

    @abstractmethod
    def reset(self) -> R:
        pass


@dataclass(frozen=True, slots=True)
class Accuracy(Metrics[float]):
    total_correct: int = 0
    total_samples: int = 0

    def update(self, predictions: np.ndarray, labels: np.ndarray) -> Accuracy:
        argmax_predictions = np.argmax(predictions, axis=1)
        argmax_labels = np.argmax(labels, axis=1)
        total_correct = self.total_correct + np.sum(argmax_predictions == argmax_labels)
        total_samples = self.total_samples + len(predictions)
        return Accuracy(total_correct, total_samples)

    def combine(self, metric: Accuracy) -> Accuracy:
        total_correct = self.total_correct + metric.total_correct
        total_samples = self.total_samples + metric.total_samples
        return Accuracy(total_correct, total_samples)

    def result(self) -> float:
        if self.total_samples == 0:
            return 0.0

        return self.total_correct / self.total_samples

    def reset(self) -> Accuracy:
        return Accuracy()

    def __str__(self):
        return f"accuracy: {self.result()}"

    def __repr__(self):
        return self.__str__()

