from .neuronet import DenseNetwork
from .lossfun import LossFunction
from abc import ABCMeta, abstractmethod
from os import cpu_count
from multiprocessing import Pool

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


class DataLabelBatchGenerator:

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

    def __init__(self, net: DenseNetwork, update_rule: UpdateRule, loss: LossFunction):
        self.__net = net
        self.__update_rule = update_rule
        self.__loss = loss
        self.__gradients = np.zeros(net.parameters.shape, dtype=object)

    def train_network(self, data_label_batch_generator: DataLabelBatchGenerator, epochs=5):
        processors = cpu_count()
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

    def __reset_gradients(self):
        self.__gradients = np.zeros(self.__net.parameters.shape, dtype=object)

    def __update_parameters(self):
        self.__net.parameters = self.__update_rule(self.__net.parameters, self.__gradients)
