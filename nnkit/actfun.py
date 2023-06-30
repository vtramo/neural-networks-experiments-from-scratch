from __future__ import annotations

import numpy as np
from abc import ABCMeta, abstractmethod


class ActivationFunction(object, metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, a: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, a: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(ActivationFunction):

    def __call__(self, a: float) -> float:
        return 1 / (1 + np.exp(-a))

    def derivative(self, a: float) -> float:
        sigma_a = self(a)
        return sigma_a * (1 - sigma_a)


class Identity(ActivationFunction):

    def __call__(self, a: float) -> float:
        return a

    def derivative(self, a: float) -> float:
        return 1


class Softmax(ActivationFunction):

    def __call__(self, a: np.ndarray) -> np.ndarray:
        a -= np.max(a)
        exp_a = np.exp(a)
        return exp_a / np.sum(exp_a)

    def derivative(self, a: float) -> float:
        raise NotImplementedError


class ReLU(ActivationFunction):

    def __call__(self, a: float) -> float:
        return max(0.0, a)

    def derivative(self, a: float) -> float:
        if a == 0:
            raise ArithmeticError("The derivative of the ReLU function is not defined at a=0")

        return 0 if a < 0 else 1
