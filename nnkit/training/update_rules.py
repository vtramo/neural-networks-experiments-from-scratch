from __future__ import annotations

from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

import copy
import numpy as np


class UpdateRule(object, metaclass=ABCMeta):

    def __init__(self, learning_rate: float = 0.1):
        self._learning_rate = learning_rate

    @abstractmethod
    def __call__(self, train_data: TrainData) -> np.ndarray:
        pass


class SGD(UpdateRule):

    def __init__(self, learning_rate: float = 0.1, momentum: float = 0.0):
        super().__init__(learning_rate)

    def __call__(self, train_data: TrainData) -> np.ndarray:
        return train_data.parameters - self._learning_rate * train_data.gradients


class RProp(UpdateRule):

    def __init__(
        self,
        initial_step_size: float = 0.01,
        increase_factor: float = 1.2,
        decrease_factor: float = 0.5,
        min_step_size: float = 1e-6,
        max_step_size: float = 50
    ):
        super().__init__(1.0)
        self._initial_stepsize = initial_step_size
        self._increase_factor = increase_factor
        self._decrease_factor = decrease_factor
        self._min_stepsize = min_step_size
        self._max_stepsize = max_step_size
        self._stepsizes_parameters = None
        self._stepsizes = None
        self.prev_gradients = None
        self._gradients_change = None

    @property
    def gradients_change(self):
        return self._gradients_change

    @dataclass(slots=True)
    class StepsizeParameter:
        stepsize: float
        value: float

        def min_max_stepsize(self, increase_factor, max_stepsize) -> float:
            self.stepsize = min(self.stepsize * increase_factor, max_stepsize)
            return self.stepsize

        def max_min_stepsize(self, decrease_factor, min_stepsize) -> float:
            self.stepsize = max(self.stepsize * decrease_factor, min_stepsize)
            return self.stepsize

        @staticmethod
        def max_min_stepsize_ndarray(weights: np.ndarray, decrease_factor, min_stepsize) -> np.ndarray:
            return np.array([w.max_min_stepsize(decrease_factor, min_stepsize) for w in weights])

        @staticmethod
        def min_max_stepsize_ndarray(weights: np.ndarray, increase_factor, max_stepsize) -> np.ndarray:
            return np.array([w.min_max_stepsize(increase_factor, max_stepsize) for w in weights])

    def __call__(self, train_data: TrainData) -> np.ndarray:
        parameters = train_data.parameters
        gradients = train_data.gradients

        if self._stepsizes is None:
            self.__init_stepsizes(parameters)
        self.__init_stepsizes_parameters(parameters)

        if self.prev_gradients is not None:
            self._gradients_change = np.array([
                np.sign(layer_gradients * prev_layer_gradients)
                for layer_gradients, prev_layer_gradients in zip(gradients, self.prev_gradients)
            ], dtype=object)

            pairs = zip(self._stepsizes_parameters, self._gradients_change)
            for i, (stepsizes_layer, gradients_change_layer) in enumerate(pairs):
                self._stepsizes[i] = self.__compute_stepsizes_layer(stepsizes_layer, gradients_change_layer)

        self.prev_gradients = gradients

        gradients_sign = np.array([np.sign(layer_gradients) for layer_gradients in gradients], dtype=object)
        delta_parameters = self.compute_delta_parameters(gradients_sign)
        return parameters + delta_parameters

    def __init_stepsizes(self, parameters: np.ndarray) -> None:
        self._stepsizes = np.array([
            np.full_like(layer_parameters, self._initial_stepsize)
            for layer_parameters in parameters
        ], dtype=object)

    def __init_stepsizes_parameters(self, parameters: np.ndarray) -> None:
        add_stepsize = np.vectorize(lambda stepsize, weight: self.StepsizeParameter(stepsize, weight))

        def get_layer_stepsizes(layer_index: int) -> float | list[list[float]]:
            if self.prev_gradients is None:
                return self._initial_stepsize
            else:
                return np.array([[w.stepsize for w in weights] for weights in self._stepsizes_parameters[layer_index]])

        self._stepsizes_parameters = np.array([
            add_stepsize(get_layer_stepsizes(layer_index), layer_parameters)
            for layer_index, layer_parameters in enumerate(parameters)
        ], dtype=object)

    def __compute_stepsizes_layer(
        self,
        prev_stepsize_layer: np.ndarray,
        gradients_change_layer: np.ndarray
    ) -> np.ndarray:
        return np.piecewise(
            prev_stepsize_layer,
            [gradients_change_layer == 1, gradients_change_layer == -1],
            [self._min_max_step_size_weights, self._max_min_step_size_weights]
        )

    def _min_max_step_size_weights(self, layer_weights: np.ndarray) -> np.ndarray:
        return self.StepsizeParameter.min_max_stepsize_ndarray(layer_weights, self._increase_factor, self._max_stepsize)

    def _max_min_step_size_weights(self, layer_weights: np.ndarray) -> np.ndarray:
        return self.StepsizeParameter.max_min_stepsize_ndarray(layer_weights, self._decrease_factor, self._min_stepsize)

    def compute_delta_parameters(self, gradients_sign: np.ndarray) -> np.ndarray:
        return -gradients_sign * self._stepsizes


class RPropPlus(RProp):

    def __init__(
        self,
        initial_step_size: float = 0.01,
        increase_factor: float = 1.2,
        decrease_factor: float = 0.5,
        min_step_size: float = 1e-6,
        max_step_size: float = 50
    ):
        super().__init__(initial_step_size, increase_factor, decrease_factor, min_step_size, max_step_size)
        self._prev_delta_parameters = None

    def compute_delta_parameters(self, gradients_sign: np.ndarray) -> np.ndarray:
        delta_parameters = numpy_deep_zeros_like(gradients_sign)

        if self._prev_delta_parameters is None:
            self._prev_delta_parameters = numpy_deep_zeros_like(gradients_sign)

        for i, layer_parameters in enumerate(delta_parameters):
            for j, neuron_parameters in enumerate(layer_parameters):
                for k in range(len(neuron_parameters)):
                    delta_parameters[i][j][k] = self.compute_delta_parameter((i, j, k), gradients_sign)

        self._prev_delta_parameters = delta_parameters
        return delta_parameters

    def compute_delta_parameter(self, indexes: tuple[int, int, int], gradients_sign: np.ndarray) -> float:
        (i, j, k) = indexes

        if self.gradients_change is not None and self.gradients_change[i][j][k] < 0:
            delta_parameter = -self._prev_delta_parameters[i][j][k]
            self.prev_gradients[i][j][k] = 0
        else:
            delta_parameter = -np.sign(gradients_sign[i][j][k]) * self._stepsizes[i][j][k]

        return delta_parameter


class IRPropPlus(RPropPlus):

    def __init__(
        self,
        initial_step_size: float = 0.01,
        increase_factor: float = 1.2,
        decrease_factor: float = 0.5,
        min_step_size: float = 1e-6,
        max_step_size: float = 50
    ):
        super().__init__(initial_step_size, increase_factor, decrease_factor, min_step_size, max_step_size)
        self._loss = 0.0
        self._prev_loss = 0.0

    def __call__(self, train_data: TrainData):
        self._loss = train_data.loss
        updated_parameters = super().__call__(train_data)
        self._prev_loss = self._loss
        return updated_parameters

    def compute_delta_parameter(self, indexes: tuple[int, int, int], gradients_sign: np.ndarray) -> float:
        (i, j, k) = indexes

        if self.gradients_change is not None and self.gradients_change[i][j][k] < 0:
            if self._prev_loss is None or self._loss > self._prev_loss:
                delta_parameter = -self._prev_delta_parameters[i][j][k]
            else:
                delta_parameter = 0
            self.prev_gradients[i][j][k] = 0
        else:
            delta_parameter = -np.sign(gradients_sign[i][j][k]) * self._stepsizes[i][j][k]

        return delta_parameter


class IRPropMinus(IRPropPlus):
    
    def compute_delta_parameter(self, indexes: tuple[int, int, int], gradients_sign: np.ndarray) -> float:
        (i, j, k) = indexes

        if self.gradients_change is not None and self.gradients_change[i][j][k] < 0:
            self.prev_gradients[i][j][k] = 0
        
        delta_parameter = -np.sign(gradients_sign[i][j][k]) * self._stepsizes[i][j][k]

        return delta_parameter

def numpy_deep_zeros_like(complex_ndarray: np.ndarray, dtype=object) -> np.ndarray:
    return np.array([
        np.zeros_like(array)
        for array in complex_ndarray
    ], dtype=dtype)

