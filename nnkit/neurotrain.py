from __future__ import annotations

from .neuronet import DenseNetwork
from .lossfun import LossFunction
from abc import ABCMeta, abstractmethod
from os import cpu_count
from multiprocessing import Pool
from typing import Generic, TypeVar

import nnkit
import numpy as np
import math


class UpdateRule(object, metaclass=ABCMeta):

    def __init__(self, learning_rate: float = 0.1):
        self._learning_rate = learning_rate

    @abstractmethod
    def __call__(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        pass


class SGD(UpdateRule):

    def __init(self, learning_rate: float = 0.1, momentum: float = 0.0):
        super().__init(learning_rate)

    def __call__(self, parameters: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        return parameters - self._learning_rate * gradients


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


    def __init__(self, points, labels, batch_size=128):
        assert len(points) == len(labels)
        self.points = points
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(points) / batch_size)

    def __iter__(self):
        return self.DataLabelIterator(self)

    class DataLabelIterator:

        def __init__(self, outer_instance):
            self.index = 0
            self.outer_instance = outer_instance

        def __next__(self):
            points = self.outer_instance.points[self.index: self.index + self.outer_instance.batch_size]
            labels = self.outer_instance.labels[self.index: self.index + self.outer_instance.batch_size]

            if len(points) == 0:
                raise StopIteration

            self.index += self.outer_instance.batch_size
            return points, labels


class NeuralNetworkSupervisedTrainer:

    def __init__(self, net: DenseNetwork, update_rule: UpdateRule, loss: LossFunction, *metrics: Metrics...):
        self.__net = net
        self.__update_rule = update_rule
        self.__loss = loss
        self.__gradients = np.zeros(net.parameters.shape, dtype=object)
        self.__metrics = list(metrics)

    def train_network(self, data_label_batch_generator: DataLabelBatchGenerator, validation_points: np.ndarray, validation_labels: np.ndarray, epochs=5):
        processors = cpu_count()
        best_parameters = (0.0, self.__net.parameters)
        extra_metrics = [0.0 for _ in range(len(self.__metrics))]

        for epoch in range(epochs):
            self.__reset_gradients()

            for points, labels in data_label_batch_generator:
                points_chunks = nnkit.fair_divide(points, processors)
                labels_chunks = nnkit.fair_divide(labels, processors)

                with Pool(processors) as pool:
                    backprop_args = [(self.__loss, points_chunks[i], labels_chunks[i]) for i in range(0, processors)]
                    for gradients in pool.starmap(self.__net.compute_gradients, backprop_args):
                        self.__gradients += gradients

                self.__update_parameters()

            metric = self.__validate_network(validation_points, validation_labels)
            if metric[0] > best_parameters[0]:
                best_parameters = (metric[0], self.__net.parameters)
                extra_metrics = metric[1:]
        
        self.__net.parameters = best_parameters[1]

        return extra_metrics
        

    def __validate_network(self, validation_points: np.ndarray, validation_labels: np.ndarray):
        processors = cpu_count()
        points_chunks = nnkit.fair_divide(validation_points, processors)
        labels_chunks = nnkit.fair_divide(validation_labels, processors)

        with Pool(processors) as pool:
            validate_iteration_args = [(points_chunks[i], labels_chunks[i]) for i in range(0, processors)]
            metrics += pool.starmap(self.__validate_iteration, validate_iteration_args)
               
        return metrics / processors
    

    def __validate_iteration(self, point_chunk: np.ndarray, label_chunk: np.ndarray):
        computed_loss = np.array([])
        for point, label in zip(point_chunk, label_chunk):
            prediction = self.__net.forward(point)
            computed_loss = np.append(computed_loss, self.__loss(prediction, label))
            for metric in self.__metrics:
                metric.update(prediction, label)
        
        computed_metrics = np.array([metric.result() for metric in self.__metrics])
        computed_loss_mean = np.mean(computed_loss)

        return np.concatenate(computed_loss_mean, computed_metrics)


    def __reset_gradients(self):
        self.__gradients = np.zeros(self.__net.parameters.shape, dtype=object)

    def __update_parameters(self):
        self.__net.parameters = self.__update_rule(self.__net.parameters, self.__gradients)
