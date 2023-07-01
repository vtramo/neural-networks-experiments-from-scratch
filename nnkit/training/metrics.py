from __future__ import annotations
from multiprocessing import Pool

from typing import Generic, TypeVar
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from joblib import cpu_count

import numpy as np

from nnkit.datasets.utils import DataLabelSet
from nnkit.lossfun import LossFunction
from nnkit.neuronet import DenseNetwork
from nnkit.training.metrics import Metrics

R = TypeVar('R')


class MetricsEvaluator():
    def __init__(self, metrics: Metrics, dataset: DataLabelSet, net: DenseNetwork, loss: LossFunction):
        self.__metrics = metrics
        self.__dataset = dataset
        self.__loss = loss
        self.__net = net
    
    def compute_metrics(self) -> MetricResults:
        processors = cpu_count()

        parallelism_allowed = processors > 1 and len(self.__dataset) >= processors * (processors / 2)
        if parallelism_allowed:
            return self.__compute_metrics_in_parallel()
        else:
            (points, labels) = self.__dataset.get()
            predictions = self.__net(points)
            loss = self.__loss(predictions, labels)
            self.__metrics = [metric.update(predictions, labels) for metric in self.__metrics]

            return self.__get_metric_results(loss)

    def __compute_metrics_in_parallel(self) -> MetricResults:
        processors = cpu_count()
        (points_chunks, labels_chunks) = self.__dataset.fair_divide(processors)

        per_chunks_losses = [0.0] * processors
        with Pool(processors) as pool:
            compute_iteration_args = [(points_chunks[i], labels_chunks[i]) for i in range(0, processors)]
            results_metrics_iterations = pool.starmap(self._compute_iteration, compute_iteration_args)
            for i, (chunk_loss, chunk_metrics) in enumerate(results_metrics_iterations):
                per_chunks_losses[i] = chunk_loss
                self.__combine_metrics(chunk_metrics)

        loss_value = np.mean(per_chunks_losses)
        return self.__get_metric_results(loss_value)

    def _compute_iteration(self, points: np.ndarray, labels: np.ndarray) -> tuple[float, list[Metrics]]:
        predictions = self.__net(points)
        loss = self.__loss(predictions, labels)
        updated_metrics = [metric.update(predictions, labels) for metric in self.__metrics]
        return np.mean(loss), updated_metrics

    def __combine_metrics(self, metrics: list[Metrics]) -> None:
        self.__metrics = [
            metric1.combine(metric2)
            for metric1, metric2 in zip(self.__metrics, metrics)
        ]
    
    def __get_metric_results(self, loss_value) -> MetricResults:
        metric_results = MetricResults()
        tmp_loss = Loss()
        tmp_loss.update(loss_value)
        metric_results.add(Loss())
        
        for metric in self.__metrics:
            metric_results.add(metric.name, metric)

        return metric_results

class MetricResults():
    def __init__(self):
        self.metrics = None

    def get(self, name: str) -> Metrics:
        return self.metrics[name]

    def add(self, metric: Metrics) -> None:
        self.metrics[metric.name()] = metric

    def __str__(self):
        return "\n".join([str(metric) for metric in self.metrics.values])

    def __repr__(self):
        return self.__str__()

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

    @abstractmethod
    def reset(self) -> R:
        pass


@dataclass(frozen=True, slots=True)
class Loss(Metrics[float]):

    @property
    def name(self) -> str:
        return "loss"

    def result(self) -> float:
        return self.__value
    
    def combine(self, metric: Metrics) -> Metrics:
        #not implemented
        pass
    
    def update(self, loss_value) -> Loss:
        self.__value = loss_value

    def reset(self) -> Loss:
        return Loss()

    def __str__(self):
        return f"{self.__name}: {self.result()}"

    def __repr__(self):
        return self.__str__()


@dataclass(frozen=True, slots=True)
class Accuracy(Metrics[float]):
    total_correct: int = 0
    total_samples: int = 0


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

    def reset(self) -> Accuracy:
        return Accuracy()
    
    @property
    def nome(self) -> str:
        return "accuracy"

    def __str__(self):
        return f"{self.__nome}: {self.result()}"

    def __repr__(self):
        return self.__str__()

