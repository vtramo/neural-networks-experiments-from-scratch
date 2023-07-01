from __future__ import annotations

from nnkit.neuronet import DenseNetwork
from nnkit.lossfun import LossFunction
from nnkit.training.update_rules import UpdateRule
from nnkit.training.metrics import MetricResults, Metrics, MetricsEvaluator
from nnkit.datasets.utils import fair_divide, DataLabelSet, DataLabelBatchGenerator

from os import cpu_count
from multiprocessing import Pool
from dataclasses import dataclass

import copy
import numpy as np


@dataclass(slots=True, frozen=True)
class TrainData:
    gradients: np.ndarray
    parameters: np.ndarray
    loss: float


class NetworkTrainer:

    def __init__(self, net: DenseNetwork, update_rule: UpdateRule, loss: LossFunction, metrics: list[Metrics]):
        self.__net = net
        self.__update_rule = update_rule
        self.__loss = loss
        self.__gradients = np.zeros(net.parameters.shape, dtype=object)
        self.__metrics = metrics
        self.__last_loss_value = 0.0

    @dataclass(slots=True)
    class ParametersWithMetrics:
        parameters: np.ndarray
        metric_results: MetricResults

        def __post_init__(self) -> None:
            self.parameters = copy.deepcopy(self.parameters)

        def set(self, parameters: np.ndarray, metric_result: MetricResults) -> None:
            self.parameters = copy.deepcopy(parameters)
            self.metric_results = metric_result

    def train_network(
        self,
        training_set_batch_generator: DataLabelBatchGenerator,
        validation_set: DataLabelSet,
        epochs=5
    ) -> ParametersWithMetrics:
        
        metrics_results = []

        best_parameters = self.ParametersWithMetrics(self.__net.parameters, metrics_results)
        metrics_evaluator = MetricsEvaluator(self.__metrics, validation_set, self.__net, self.__loss)    

        for epoch in range(epochs):

            for points, labels in training_set_batch_generator:
                self.__compute_gradients(points, labels)
                self.__update_parameters()
                self.__reset_gradients()

            metric_result = metrics_evaluator.compute_metrics()
            metrics_results.append(metric_result)
            self.__last_loss_value = metric_result.get("loss")
            if self.__last_loss_value < best_parameters.metric_results.get("loss"):
                best_parameters.set(self.__net.parameters, metric_result)

            print(metric_result)
            self.__reset_metrics()

        self.__net.parameters = best_parameters.parameters

        return best_parameters

    def __reset_gradients(self) -> None:
        self.__gradients = np.zeros(self.__net.parameters.shape, dtype=object)

    def __reset_metrics(self) -> None:
        self.__metrics = [metric.reset() for metric in self.__metrics]

    def __compute_gradients(self, points: np.ndarray, labels: np.ndarray) -> None:
        processors = cpu_count()

        parallelism_allowed = processors > 1 and len(points) >= processors * (processors / 2)
        if parallelism_allowed:
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
        gradients = copy.deepcopy(self.__gradients)
        parameters = copy.deepcopy(self.__net.parameters)
        train_data = TrainData(gradients, parameters, self.__last_loss_value)

        self.__net.parameters = self.__update_rule(train_data)
