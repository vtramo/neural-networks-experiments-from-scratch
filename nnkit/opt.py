from abc import ABCMeta, abstractmethod
from .neuronet import DenseNetwork


class UpdateRule(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, parameters: numpy.array, learning_rate: float = 10):
        self.__parameters = parameters
        self.__learning_rate = learning_rate

    def __call__(self) -> numpy.array:
        raise NotImplementedError


class Optimizer:
    def __init__(self, net: DenseNetwork, update_rule: UpdateRule):
        self.__net = net
        self.__update_rule = update_rule

    def __call__(self, x: numpy.array, t: numpy.array) -> numpy.array:
        return self.__optimize(x, t)

    def __optimize(self, x: numpy.array, t: numpy.array) -> numpy.array:
        pass


def back_prop():
    pass
