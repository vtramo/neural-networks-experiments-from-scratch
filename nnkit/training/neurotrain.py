from __future__ import annotations

from nnkit.neuronet import DenseNetwork
from nnkit.lossfun import LossFunction
from nnkit.datasets.utils import fair_divide
from abc import ABCMeta, abstractmethod
from os import cpu_count
from multiprocessing import Pool
from dataclasses import dataclass
from collections.abc import Callable

import os
import nnkit
import copy
import numpy as np
import math


class NetworkTrainer:

    def __init__(self, net: DenseNetwork, update_rule: UpdateRule, loss: LossFunction, metrics: list[Metrics]):
        self.__net = net
        self.__update_rule = update_rule
        self.__loss = loss
        self.__gradients = np.zeros(net.parameters.shape, dtype=object)
        self.__metrics = metrics

    @dataclass(slots=True)
    class ParametersWithMetrics:
        parameters: np.ndarray
        loss: float = float('inf')
        extra_metrics: list[Metrics] = None

        def __post_init__(self) -> None:
            self.parameters = copy.deepcopy(self.parameters)

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

        best_parameters = self.ParametersWithMetrics(self.__net.parameters)

        for epoch in range(epochs):
            self.__reset_gradients()

            for points, labels in training_set_batch_generator:
                self.__compute_gradients(points, labels)
                self.__update_parameters()

            loss = self.__validate_network(validation_set)
            if loss < best_parameters.loss:
                best_parameters.set(self.__net.parameters, loss, self.__metrics)

            self.__print_epoch_info(epoch, loss)

        self.__net.parameters = best_parameters.parameters

        return best_parameters

    def __reset_gradients(self) -> None:
        self.__gradients = np.zeros(self.__net.parameters.shape, dtype=object)

    def __compute_gradients(self, points: np.ndarray, labels: np.ndarray) -> None:
        processors = cpu_count()

        if len(points) >= processors * (processors / 2):
            self.__compute_gradients_in_parallel(points, labels)
        else:
            self.__gradients += self.__net.compute_gradients(self.__loss, points, labels)

    def __compute_gradients_in_parallel(self, points: np.ndarray, labels: np.ndarray) -> None:
        processors = cpu_count()
        (points_chunks, labels_chunks) = fair_divide(points, labels, workers=processors)

        with Pool(processors) as pool:
            backprop_args = [(self.__loss, points_chunks[i], labels_chunks[i]) for i in range(0, processors)]
            for gradients in pool.starmap(self.__net.compute_gradients, backprop_args):
                self.__gradients += gradients

    def __update_parameters(self) -> None:
        self.__net.parameters = self.__update_rule(self.__net.parameters, self.__gradients)

    def __validate_network(self, validation_set: DataLabelSet) -> float:
        processors = cpu_count()

        if len(validation_set) >= processors * (processors / 2):
            return self.__validate_network_in_parallel(validation_set)
        else:
            (points, labels) = validation_set.get()
            predictions = self.__net(points)
            loss = self.__loss(predictions, labels)
            self.__metrics = [metric.update(predictions, labels) for metric in self.__metrics]
            return np.mean(loss)

    def __validate_network_in_parallel(self, validation_set: DataLabelSet) -> float:
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
        return np.mean(loss), updated_metrics

    def __combine_metrics(self, metrics: Metrics) -> None:
        self.__metrics = [metric.combine(metrics[i]) for i, metric in enumerate(self.__metrics)]

    def __print_epoch_info(self, epoch, loss) -> None:
        metrics_info = [f"loss: {loss}"] + [str(metric) for metric in self.__metrics]
        print(f"Epoch {epoch}: {metrics_info}")
