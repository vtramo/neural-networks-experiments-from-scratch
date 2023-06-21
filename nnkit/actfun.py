import numpy as np
from abc import ABCMeta, abstractmethod


class ActivationFunction(object, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(ActivationFunction):

    def __call__(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: float) -> float:
        sigma_x = self(x)
        return sigma_x * (1 - sigma_x)


class Identity(ActivationFunction):

    def __call__(self, x: float) -> float:
        return x

    def derivative(self, x: float) -> float:
        return 1


class Softmax(ActivationFunction):

    def __call__(self, y: np.ndarray) -> np.ndarray:
        exp_y = np.exp(y)
        exp_sum_y = np.sum(exp_y)
        return np.array([
            np.exp(y_value) / exp_sum_y
            for y_value in y
        ])

    def derivative(self, x: float) -> float:
        raise NotImplementedError
