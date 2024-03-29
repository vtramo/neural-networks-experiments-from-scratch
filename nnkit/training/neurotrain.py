from __future__ import annotations

from nnkit.core.neuronet import DenseNetwork
from nnkit.core.losses import LossFunction
from nnkit.training.update_rules import UpdateRule
from nnkit.training.metrics import MetricResults, Metrics, MetricsEvaluator
from nnkit.datasets.utils import fair_divide, DataLabelSet

from os import cpu_count
from multiprocessing import Pool
from dataclasses import dataclass, field

import copy
import numpy as np


@dataclass(slots=True, frozen=True)
class TrainingData:
    gradients: np.ndarray = None
    parameters: np.ndarray = None
    val_loss: float = None
    train_loss: float = None


@dataclass(slots=True)
class ParametersWithMetrics:
    epoch: int
    parameters: np.ndarray
    metric_results: MetricResults = None
    loss: float = float('inf')

    def __post_init__(self) -> None:
        self.parameters = copy.deepcopy(self.parameters)


@dataclass(slots=True)
class TrainingHistory:
    epochs: int
    steps: int
    update_rule: str = ""
    best_parameters: ParametersWithMetrics = None
    history: list[dict[str, Metrics]] = field(default_factory=lambda: [None])

    def __post_init__(self):
        self.history = [None] * self.epochs

    def store(self, *results: MetricResults, epoch: int) -> None:
        for metric_results in results:
            metric_by_name = metric_results.metric_by_name
            if self.history[epoch] is None:
                self.history[epoch] = metric_by_name
            else:
                self.history[epoch].update(metric_by_name)


class NetworkTrainer:

    def __init__(
        self,
        net: DenseNetwork,
        update_rule: UpdateRule,
        loss_function: LossFunction,
        metrics: list[Metrics],
        multiprocessing: bool = True,
        metrics_evaluator: MetricsEvaluator = None
    ):
        self.__net = net
        self.__update_rule = update_rule
        self.__loss_function = loss_function
        self.__gradients = np.zeros(net.parameters.shape, dtype=object)
        self.__last_val_loss = 0.0
        self.__best_parameters = ParametersWithMetrics(parameters=self.__net.parameters, epoch=0)
        self.__multiprocessing = multiprocessing
        self.__train_history: TrainingHistory = None
        self.__current_epoch = 0
        if metrics_evaluator is not None:
            self.__metrics_evaluator = metrics_evaluator
        else:
            self.__metrics_evaluator = MetricsEvaluator(self.__net, metrics, self.__loss_function)

    def train_network(
        self,
        training_set: DataLabelSet,
        validation_set: DataLabelSet,
        epochs: int,
        early_stopping: function[TrainingData, bool] | nnkit.training.stopping.StoppingCriterion = lambda _: False
    ) -> TrainingHistory:

        self.__train_history = TrainingHistory(
            epochs,
            steps=training_set.steps,
            update_rule=self.__update_rule.name
        )

        for epoch in range(epochs):
            self.__current_epoch = epoch
            for points, labels in training_set:
                self.__compute_gradients(points, labels)
                self.__update_parameters()
                self.__reset_gradients()

            val_metric_results, train_metric_results = self.__evaluate_net(validation_set, training_set)
            self.__update_best_parameters(val_metric_results)
            self.__print_epoch_info()

            train_data = TrainingData(val_loss=self.__last_val_loss, train_loss=self.__last_train_loss)
            if early_stopping(train_data):
                print("Stopped early!")
                break

        self.__net.parameters = copy.deepcopy(self.__best_parameters.parameters)
        history = self.__train_history
        self.__reset_trainer()
        return history

    def __compute_gradients(self, points: np.ndarray, labels: np.ndarray) -> None:
        if self.__is_multiprocessing_allowed(num_samples_dataset=len(points)):
            self.__compute_gradients_in_parallel(points, labels)
        else:
            self.__gradients += self.__net.compute_gradients(self.__loss_function, points, labels)

    def __is_multiprocessing_allowed(self, num_samples_dataset: int) -> bool:
        processors = cpu_count()
        return self.__multiprocessing and processors > 1 and num_samples_dataset >= processors * (processors / 2)

    def __compute_gradients_in_parallel(self, points: np.ndarray, labels: np.ndarray) -> None:
        processors = cpu_count()
        (points_chunks, labels_chunks) = fair_divide(points, labels, workers=processors)

        with Pool(processors) as pool:
            backprop_args = [(self.__loss_function, points_chunks[i], labels_chunks[i]) for i in range(0, processors)]
            for gradients in pool.starmap(self.__net.compute_gradients, backprop_args):
                self.__gradients += gradients

    def __update_parameters(self) -> None:
        gradients = copy.deepcopy(self.__gradients)
        parameters = copy.deepcopy(self.__net.parameters)
        train_data = TrainingData(gradients, parameters, val_loss=self.__last_val_loss)

        self.__net.parameters = self.__update_rule(train_data)

    def __reset_gradients(self) -> None:
        self.__gradients = np.zeros(self.__net.parameters.shape, dtype=object)

    def __evaluate_net(
        self,
        val_set: DataLabelSet,
        train_set: DataLabelSet,
        *data_sets: DataLabelSet
    ) -> tuple[MetricResults, MetricResults]:

        validation_metric_results = self.__metrics_evaluator.compute_metrics(val_set)
        validation_loss_metric = validation_metric_results[self.__loss_function.name]
        self.__last_val_loss = validation_loss_metric.result()

        validation_metric_prefix = "validation" if not val_set.name else val_set.name
        validation_metric_results.prefix(validation_metric_prefix)
        self.__train_history.store(validation_metric_results, epoch=self.__current_epoch)

        training_metric_results = self.__metrics_evaluator.compute_metrics(train_set)
        training_loss_metric = training_metric_results[self.__loss_function.name]
        self.__last_train_loss = training_loss_metric.result()

        training_metric_prefix = "training" if not train_set.name else train_set.name
        training_metric_results.prefix(training_metric_prefix)
        self.__train_history.store(training_metric_results, epoch=self.__current_epoch)

        for data_set in data_sets:
            metric_results = self.__metrics_evaluator.compute_metrics(data_set)
            metric_results.prefix(data_set.name)
            self.__train_history.store(metric_results, epoch=self.__current_epoch)

        return validation_metric_results, training_metric_results

    def __update_best_parameters(self, val_metric_results: MetricResults) -> None:
        if self.__last_val_loss >= self.__best_parameters.loss:
            return

        self.__best_parameters.parameters = copy.deepcopy(self.__net.parameters)
        self.__best_parameters.metric_results = val_metric_results
        self.__best_parameters.loss = self.__last_val_loss
        self.__best_parameters.epoch = self.__current_epoch
        self.__train_history.best_parameters = self.__best_parameters

    def __print_epoch_info(self) -> None:
        last_metric_results = self.__train_history.history[self.__current_epoch]
        epoch_info = f"{[str(metric_result) for metric_result in last_metric_results.values()]}"
        print(f"Epoch {self.__current_epoch + 1} - {epoch_info}")

    def __reset_trainer(self):
        self.__gradients = np.zeros(self.__net.parameters.shape, dtype=object)
        self.__last_val_loss = 0.0
        self.__current_epoch = 0
        self.__best_parameters = ParametersWithMetrics(parameters=self.__net.parameters, epoch=0)
        self.__train_history: TrainingHistory = None

    @property
    def net(self) -> DenseNetwork:
        return self.__net
