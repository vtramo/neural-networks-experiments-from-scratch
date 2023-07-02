from __future__ import annotations

from nnkit.datasets.utils import DataLabelSet
from nnkit.losses import LossFunction
from nnkit.neuronet import DenseNetwork

from multiprocessing import Pool
from typing import Generic, TypeVar
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from os import cpu_count

import copy
import numpy as np


R = TypeVar('R')


class Metrics(Generic[R], metaclass=ABCMeta):

    def __init__(self, name: str = ""):
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

    @abstractmethod
    def update(self, predictions: np.ndarray, label: np.ndarray) -> Metrics:
        pass

    @abstractmethod
    def combine(self, metric: Metrics) -> Metrics:
        pass

    @abstractmethod
    def result(self) -> R:
        pass

    @abstractmethod
    def reset(self) -> R:
        pass

    def __str__(self):
        return f"{self.name}: {self.result()}"


@dataclass(slots=True)
class MetricResults:
    metrics: list[Metrics]
    metric_by_name: dict[str, Metrics] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        self.metric_by_name = {metric.name: metric for metric in self.metrics}

    def __iter__(self) -> Iterator[tuple[str, Metrics]]:
        return iter(self.metric_by_name.items())

    def __getitem__(self, metric_name: str) -> float | Metrics:
        if type(metric_name) is not str:
            raise TypeError("metric_name must be a string!")

        return self.metric_by_name[metric_name]

    def __str__(self):
        return " - ".join([str(metric) for metric in self.metrics])

    def prefix(self, prefix: str):
        prefixed_metric_by_name = {}
        for metric_name, metric in self.metric_by_name.items():
            prefixed_metric_name = f"{prefix}_{metric_name}"
            metric.name = prefixed_metric_name
            prefixed_metric_by_name[prefixed_metric_name] = self.metric_by_name[metric_name]
        self.metric_by_name = prefixed_metric_by_name


class MetricsEvaluator:

    def __init__(
        self,
        net: DenseNetwork,
        metrics: list[Metrics],
        loss_function: LossFunction,
        multiprocessing: bool = True
    ):
        self.__net = net
        self.__metrics = [Loss(loss_function, name=loss_function.name)] + metrics
        self.__multiprocessing = multiprocessing

    def compute_metrics(self, dataset: DataLabelSet) -> MetricResults:
        if self.__is_multiprocessing_allowed(num_samples_dataset=len(dataset)):
            self.__compute_metrics_in_parallel(dataset)
        else:
            (points, labels) = dataset.get()
            predictions = self.__net(points)
            self.__metrics = self.__update_metrics(predictions, labels)

        metrics = copy.copy(self.__metrics)
        self.__reset_metrics()
        return MetricResults(metrics)

    def __is_multiprocessing_allowed(self, num_samples_dataset: int) -> bool:
        processors = cpu_count()
        return self.__multiprocessing and processors > 1 and num_samples_dataset >= processors * (processors / 2)

    def __compute_metrics_in_parallel(self, dataset: DataLabelSet) -> None:
        processors = cpu_count()
        (points_chunks, labels_chunks) = dataset.fair_divide(processors)

        with Pool(processors) as pool:
            compute_metrics_args = [(points_chunks[i], labels_chunks[i]) for i in range(0, processors)]
            metric_computation_results = pool.starmap(self._compute_metrics, compute_metrics_args)
            for i, chunk_metrics in enumerate(metric_computation_results):
                self.__combine_metrics(chunk_metrics)

    def __update_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> list[Metrics]:
        return [metric.update(predictions, labels) for metric in self.__metrics]

    def _compute_metrics(self, points: np.ndarray, labels: np.ndarray) -> list[Metrics]:
        predictions = self.__net(points)
        updated_metrics = self.__update_metrics(predictions, labels)
        return updated_metrics

    def __combine_metrics(self, metrics: list[Metrics]) -> None:
        self.__metrics = [
            metric1.combine(metric2)
            for metric1, metric2 in zip(self.__metrics, metrics)
        ]

    def __reset_metrics(self) -> None:
        self.__metrics = [metric.reset() for metric in self.__metrics]


class Loss(Metrics[float]):

    def __init__(self, loss_function: LossFunction, losses: np.ndarray = np.empty(0), name: str = ""):
        super().__init__(name)
        self._loss_function = loss_function
        self._losses = losses

    def result(self) -> float:
        if not np.any(self._losses):
            return 0.0

        return np.mean(self._losses)

    def combine(self, metric: Loss) -> Loss:
        losses = np.concatenate((self._losses, metric._losses))
        return Loss(self._loss_function, losses, name=self.name)

    def update(self, predictions: np.ndarray, labels: np.ndarray) -> Loss:
        losses = np.concatenate((self._loss_function(predictions, labels), self._losses))
        return Loss(self._loss_function, losses, name=self.name)

    def reset(self) -> Loss:
        return Loss(self._loss_function, name=self.name)


class Accuracy(Metrics[float]):

    def __init__(self, total_correct: int = 0, total_samples: int = 0, name: str = ""):
        super().__init__(name)
        self._total_correct = total_correct
        self._total_samples = total_samples

    def update(self, predictions: np.ndarray, labels: np.ndarray) -> Accuracy:
        argmax_predictions = np.argmax(predictions, axis=1)
        argmax_labels = np.argmax(labels, axis=1)
        total_correct = self._total_correct + np.sum(argmax_predictions == argmax_labels)
        total_samples = self._total_samples + len(predictions)
        return Accuracy(total_correct, total_samples, name=self.name)

    def combine(self, metric: Accuracy) -> Accuracy:
        total_correct = self._total_correct + metric._total_correct
        total_samples = self._total_samples + metric._total_samples
        return Accuracy(total_correct, total_samples, self.name)

    def result(self) -> float:
        if self._total_samples == 0:
            return 0.0

        return self._total_correct / self._total_samples

    def reset(self) -> Accuracy:
        return Accuracy(name=self.name)
