import numpy as np
from abc import ABCMeta, abstractmethod


class LossFunction(object, metaclass=ABCMeta):

    def __init__(self, name: str = "loss"):
        self.name = name

    @abstractmethod
    def __call__(self, prediction: np.ndarray, gold_label: np.ndarray) -> float:
        pass

    @abstractmethod
    def output_derivative(self, prediction: np.ndarray, gold_label: np.ndarray) -> np.ndarray:
        pass


class CrossEntropy(LossFunction):

    def __init__(self, name: str = "cross_entropy_loss"):
        super().__init__(name)

    def __call__(self, predictions: np.ndarray, gold_labels: np.ndarray) -> float:
        axis = int((predictions.ndim == 2))
        return -np.sum(gold_labels * np.log(predictions), axis)

    def output_derivative(self, prediction: np.ndarray, gold_label: np.ndarray) -> np.ndarray:
        return (prediction - gold_label) / (prediction * (1 - prediction))


class CrossEntropySoftmax(CrossEntropy):

    def output_derivative(self, prediction: np.ndarray, gold_label: np.ndarray) -> np.ndarray:
        return prediction - gold_label


