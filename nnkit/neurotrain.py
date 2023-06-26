from __future__ import annotations

from .neuronet import DenseNetwork
from .lossfun import LossFunction
from abc import ABCMeta, abstractmethod
from os import cpu_count
from multiprocessing import Pool
from typing import Generic, TypeVar

import nnkit
import copy
import numpy as np
import math


class UpdateRule(object, metaclass=ABCMeta):

    def __init__(self, learning_rate: float = 0.1):
        self._learning_rate = learning_rate

    @abstractmethod
    def __call__(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        pass


class SGD(UpdateRule):

    def __init__(self, learning_rate: float = 0.1, momentum: float = 0.0):
        super().__init(learning_rate)

    def __call__(self, parameters: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        return parameters - self._learning_rate * gradients
    
class RProp(UpdateRule):

    def __init__(self, learning_rate: float = 0.01, initial_step_size: float = 0.1, increase_factor: float = 1.2, decrease_factor: float = 0.5, min_step_size: float = 1e-6, max_step_size: float = 1.0):
        super().__init__(learning_rate)
        self._initial_step_size = initial_step_size
        self._increase_factor = increase_factor
        self._decrease_factor = decrease_factor
        self._min_step_size = min_step_size
        self._max_step_size = max_step_size
        self._step_sizes = None
        self._prev_gradient = None

    def __call__(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self._step_sizes is None:
            self._step_sizes = np.full_like(parameters, self._initial_step_size)

        if self._prev_gradient is not None:
            gradient_change = np.sign(gradient * self._prev_gradient)

            if(gradient_change > 0):
                self._step_sizes = np.minimum(self._step_sizes * self._increase_factor, self._max_step_size)
            elif(gradient_change < 0):
                self._step_sizes = np.maximum(self._step_sizes * self._decrease_factor, self._min_step_size)
            else:
                self._step_sizes = self._step_sizes

        self._prev_gradient = gradient

        gradient_sign = np.where(gradient > 0, 1.0, -1.0)

        return parameters - self._learning_rate * gradient_sign * self._step_sizes


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


class Accuracy(Metrics[float]):

    def __init__(self, total_correct: int = 0, total_samples: int = 0):
        self.total_correct = total_correct
        self.total_samples = total_samples

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

    def __str__(self):
        return f"accuracy: {self.result()}"

    def __repr__(self):
        return self.__str__()


class DataLabelSet:

    def __init__(self, points, labels):
        assert len(points) == len(labels)
        self._points = points
        self._labels = labels

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        return self._points, self._labels

    def fair_divide(self, workers: int) -> tuple[list[list], list[list]]:
        return nnkit.fair_divide(self._points, self._labels, workers=workers)


class DataLabelBatchGenerator(DataLabelSet):

    def __init__(self, points, labels, batch_size=128):
        super().__init__(points, labels)
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(points) / batch_size)

    def __iter__(self):
        return self.DataLabelIterator(self)

    class DataLabelIterator:

        def __init__(self, outer_instance):
            self.index = 0
            self.outer_instance = outer_instance

        def __next__(self):
            points = self.outer_instance._points[self.index: self.index + self.outer_instance.batch_size]
            labels = self.outer_instance._labels[self.index: self.index + self.outer_instance.batch_size]

            if len(points) == 0:
                raise StopIteration

            self.index += self.outer_instance.batch_size
            return points, labels


class NetworkTrainer:

    def __init__(self, net: DenseNetwork, update_rule: UpdateRule, loss: LossFunction, metrics: list[Metrics]):
        self.__net = net
        self.__update_rule = update_rule
        self.__loss = loss
        self.__gradients = np.zeros(net.parameters.shape, dtype=object)
        self.__metrics = metrics

    class ParametersWithMetrics:
        def __init__(self, parameters: np.ndarray):
            self.parameters = copy.deepcopy(parameters)
            self.loss = float('inf')
            self.extra_metrics = []

        def set(self, parameters: np.ndarray, loss: float, extra_metrics: list[Metrics]):
            self.parameters = copy.deepcopy(parameters)
            self.loss = loss
            self.extra_metrics = extra_metrics

    def train_network(
        self,
        training_set_batch_generator: DataLabelBatchGenerator,
        validation_set: DataLabelSet,
        epochs=5
    ) -> ParametersWithMetrics:

        processors = cpu_count()
        best_parameters = self.ParametersWithMetrics(self.__net.parameters)

        for epoch in range(epochs):
            self.__reset_gradients()

            for points, labels in training_set_batch_generator:
                (points_chunks, labels_chunks) = nnkit.fair_divide(points, labels, workers=processors)

                with Pool(processors) as pool:
                    backprop_args = [(self.__loss, points_chunks[i], labels_chunks[i]) for i in range(0, processors)]
                    for gradients in pool.starmap(self.__net.compute_gradients, backprop_args):
                        self.__gradients += gradients

                self.__update_parameters()

            loss = self.__validate_network(validation_set)
            if loss < best_parameters.loss:
                best_parameters.set(self.__net.parameters, loss, self.__metrics)

            self.__print_epoch_info(epoch, loss)

        self.__net.parameters = best_parameters.parameters

        return best_parameters

    def __reset_gradients(self):
        self.__gradients = np.zeros(self.__net.parameters.shape, dtype=object)

    def __update_parameters(self):
        self.__net.parameters = self.__update_rule(self.__net.parameters, self.__gradients)

    def __validate_network(self, validation_set: DataLabelSet) -> float:
        processors = cpu_count()
        (points_chunks, labels_chunks) = validation_set.fair_divide(processors)

        per_chunks_losses = [0.0] * processors
        with Pool(processors) as pool:
            validate_iteration_args = [(points_chunks[i], labels_chunks[i]) for i in range(0, processors)]
            results_validate_iterations = pool.starmap(self._validate_iteration, validate_iteration_args)
            for i, (chunk_loss, chunk_metrics) in enumerate(results_validate_iterations):
                per_chunks_losses[i] = chunk_loss
                self.__combine_metrics(chunk_metrics)

        return np.mean(per_chunks_losses)

    def _validate_iteration(self, points: np.ndarray, labels: np.ndarray) -> tuple[float, list[Metrics]]:
        predictions = self.__net(points)
        loss = self.__loss(predictions, labels)
        updated_metrics = [metric.update(predictions, labels) for metric in self.__metrics]
        return loss, updated_metrics

    def __combine_metrics(self, metrics: Metrics):
        self.__metrics = [metric.combine(metrics[i]) for i, metric in enumerate(self.__metrics)]

    def __print_epoch_info(self, epoch, loss):
        metrics_info = [f"loss: {loss}"] + [str(metric) for metric in self.__metrics]
        print(f"Epoch {epoch}: {metrics_info}")
