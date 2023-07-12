from __future__ import annotations

from nnkit.utils import numpy_deep_apply_func, numpy_deep_zeros_like, numpy_deep_full_like
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

import numpy as np


class UpdateRule(object, metaclass=ABCMeta):

    def __init__(self, learning_rate: float = 0.1, name: str = ""):
        if not name:
            name = self.__class__.__name__.lower()

        self.name = name
        self._learning_rate = learning_rate

    @abstractmethod
    def __call__(self, train_data: TrainingData) -> np.ndarray:
        pass


class SGD(UpdateRule):

    def __init__(self, learning_rate: float, momentum: float = 0.0, name: str = ""):
        super().__init__(learning_rate, name)
        self.__momentum = momentum
        self.__prev_delta_parameters = 0.0

    def __call__(self, train_data: TrainingData) -> np.ndarray:
        parameters = train_data.parameters
        gradients = train_data.gradients

        delta_parameters = -(self._learning_rate * gradients) - (self.__momentum * self.__prev_delta_parameters)
        if self.__momentum != 0:
            self.__prev_delta_parameters = delta_parameters

        return parameters + delta_parameters


class AbstractRProp(UpdateRule, metaclass=ABCMeta):

    def __init__(
        self,
        initial_step_size: float = 0.01,
        increase_factor: float = 1.2,
        decrease_factor: float = 0.5,
        min_step_size: float = 1e-6,
        max_step_size: float = 50,
        name: str = ""
    ):
        super().__init__(1.0, name)
        self._initial_stepsize = initial_step_size
        self._increase_factor = increase_factor
        self._decrease_factor = decrease_factor
        self._min_stepsize = min_step_size
        self._max_stepsize = max_step_size
        self._stepsizes = None
        self._prev_gradients = None
        self._gradients_change = None
        self._gradients_sign = None

    def __call__(self, train_data: TrainingData) -> np.ndarray:
        parameters = train_data.parameters
        gradients = train_data.gradients
        self._gradients_sign = numpy_deep_apply_func(np.sign, gradients)

        if self._stepsizes is None:
            self._stepsizes = numpy_deep_full_like(parameters, self._initial_stepsize)

        if self._prev_gradients is not None:
            self._gradients_change = self._compute_gradients_change()
            self._stepsizes = self._compute_stepsizes()

        self._prev_gradients = gradients

        delta_parameters = self._compute_delta_parameters()
        return parameters + delta_parameters

    def _compute_gradients_change(self) -> np.ndarray:
        prev_gradients_sign = numpy_deep_apply_func(np.sign, self._prev_gradients)
        return prev_gradients_sign * self._gradients_sign

    def _compute_stepsizes(self) -> np.ndarray:
        stepsizes = numpy_deep_zeros_like(self._stepsizes)

        for i, layer_gradients_change in enumerate(self._gradients_change):
            for (j, k), gradient_change in np.ndenumerate(layer_gradients_change):
                if gradient_change == 1:
                    stepsizes[i][j][k] = min(self._stepsizes[i][j][k] * self._increase_factor, self._max_stepsize)
                elif gradient_change == -1:
                    stepsizes[i][j][k] = max(self._stepsizes[i][j][k] * self._decrease_factor, self._min_stepsize)

        return stepsizes

    def _compute_delta_parameters(self) -> np.ndarray:
        delta_parameters = numpy_deep_zeros_like(self._stepsizes)

        for i, layer_delta_parameters in enumerate(delta_parameters):
            for (j, k), _ in np.ndenumerate(layer_delta_parameters):
                delta_parameters[i][j][k] = self._compute_delta_parameter((i, j, k))

        return delta_parameters

    @abstractmethod
    def _compute_delta_parameter(self, indexes: tuple[int, int, int]) -> float:
        pass


class RProp(AbstractRProp):

    def __init__(
        self,
        initial_step_size: float = 0.01,
        increase_factor: float = 1.2,
        decrease_factor: float = 0.5,
        min_step_size: float = 1e-6,
        max_step_size: float = 50,
        name: str = ""
    ):
        super().__init__(initial_step_size, increase_factor, decrease_factor, min_step_size, max_step_size, name)
        self._delta_parameters = None

    def _compute_delta_parameters(self) -> np.ndarray:
        self._delta_parameters = super()._compute_delta_parameters()
        return self._delta_parameters

    def _compute_delta_parameter(self, indexes: tuple[int, int, int]) -> float:
        (i, j, k) = indexes

        if self._gradients_change is None or self._gradients_change[i][j][k] >= 0:
            return -self._gradients_sign[i][j][k] * self._stepsizes[i][j][k]

        self._prev_gradients = 0
        return -self._delta_parameters[i][j][k]


class RPropMinus(AbstractRProp):

    def _compute_delta_parameter(self, indexes: tuple[int, int, int]) -> float:
        (i, j, k) = indexes
        return -self._gradients_sign[i][j][k] * self._stepsizes[i][j][k]


class IRPropMinus(RPropMinus):

    def _compute_delta_parameter(self, indexes: tuple[int, int, int]) -> float:
        (i, j, k) = indexes

        if self._gradients_change is not None and self._gradients_change[i][j][k] < 0:
            self._prev_gradients[i][j][k] = 0

        return -np.sign(self._gradients_sign[i][j][k]) * self._stepsizes[i][j][k]


class RPropPlus(AbstractRProp):

    def __init__(
        self,
        initial_step_size: float = 0.01,
        increase_factor: float = 1.2,
        decrease_factor: float = 0.5,
        min_step_size: float = 1e-6,
        max_step_size: float = 50,
        name: str = ""
    ):
        super().__init__(initial_step_size, increase_factor, decrease_factor, min_step_size, max_step_size, name)
        self._prev_delta_parameters = None

    def _compute_delta_parameters(self) -> np.ndarray:
        if self._prev_delta_parameters is None:
            self._prev_delta_parameters = numpy_deep_zeros_like(self._stepsizes)

        delta_parameters = super()._compute_delta_parameters()
        self._prev_delta_parameters = delta_parameters

        return delta_parameters

    def _compute_delta_parameter(self, indexes: tuple[int, int, int]) -> float:
        (i, j, k) = indexes

        if self._gradients_change is not None and self._gradients_change[i][j][k] < 0:
            delta_parameter = -self._prev_delta_parameters[i][j][k]
            self._prev_gradients[i][j][k] = 0
        else:
            delta_parameter = -np.sign(self._gradients_sign[i][j][k]) * self._stepsizes[i][j][k]

        return delta_parameter


class IRPropPlus(RPropPlus):

    def __init__(
        self,
        initial_step_size: float = 0.01,
        increase_factor: float = 1.2,
        decrease_factor: float = 0.5,
        min_step_size: float = 1e-6,
        max_step_size: float = 50,
        name: str = ""
    ):
        super().__init__(initial_step_size, increase_factor, decrease_factor, min_step_size, max_step_size, name)
        self._loss = 0.0
        self._prev_loss = 0.0

    def __call__(self, train_data: TrainingData):
        self._loss = train_data.val_loss
        updated_parameters = super().__call__(train_data)
        self._prev_loss = self._loss
        return updated_parameters

    def _compute_delta_parameter(self, indexes: tuple[int, int, int]) -> float:
        (i, j, k) = indexes

        if self._gradients_change is not None and self._gradients_change[i][j][k] < 0:
            if self._loss > self._prev_loss:
                delta_parameter = -self._prev_delta_parameters[i][j][k]
            else:
                delta_parameter = 0
            self._prev_gradients[i][j][k] = 0
        else:
            delta_parameter = -np.sign(self._gradients_sign[i][j][k]) * self._stepsizes[i][j][k]

        return delta_parameter

