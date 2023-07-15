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

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return np.where(
            a >= 0,
            1 / (1 + np.exp(-a)),
            np.exp(a) / (1 + np.exp(a))
        )

    def derivative(self, a: np.ndarray) -> np.ndarray:
        sigma_a = self(a)
        return sigma_a * (1 - sigma_a)


class Tanh(ActivationFunction):

    def __call__(self, a: float) -> float:
        return np.tanh(a)

    def derivative(self, a: float) -> float:
        tanh_a = self(a)
        return 1 - tanh_a ** 2


class Identity(ActivationFunction):

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return a

    def derivative(self, a: np.ndarray) -> np.ndarray:
        return np.full_like(a, 1)


class Softmax(ActivationFunction):

    def __call__(self, a: np.ndarray) -> np.ndarray:
        a -= np.max(a)
        exp_a = np.exp(a)
        return exp_a / np.sum(exp_a)

    def derivative(self, a: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(ActivationFunction):

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return max(0.0, a)

    def derivative(self, a: np.ndarray) -> np.ndarray:
        if 0 in a:
            raise ArithmeticError("The derivative of the ReLU function is not defined at a=0")

        return np.where(a < 0, 0, 1)
