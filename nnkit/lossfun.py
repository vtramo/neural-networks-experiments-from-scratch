import numpy as np
from abc import ABCMeta, abstractmethod


class LossFunction(object, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, prediction: np.ndarray, gold_label: np.ndarray) -> float:
        pass

    @abstractmethod
    def output_derivative(self, prediction: np.ndarray, gold_label: np.ndarray) -> np.ndarray:
        pass


class CrossEntropy(LossFunction):

    def __call__(self, prediction: np.ndarray, gold_label: np.ndarray) -> float:
        return -np.sum(prediction * np.log(gold_label))

    def output_derivative(self, prediction: np.ndarray, gold_label: np.ndarray) -> np.ndarray:
        return (prediction - gold_label) / (prediction * (1 - prediction))


class CrossEntropySoftmax(CrossEntropy):

    def output_derivative(self, prediction: np.ndarray, gold_label: np.ndarray) -> np.ndarray:
        return prediction - gold_label


