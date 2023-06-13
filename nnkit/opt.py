from abc import ABCMeta, abstractmethod
from .neuronet import DenseNetwork
from .lossfun import LossFunction


class UpdateRule(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, parameters: numpy.array, learning_rate: float = 10):
        self.__parameters = parameters
        self.__learning_rate = learning_rate

    def __call__(self) -> numpy.array:
        raise NotImplementedError


class Optimizer:
    def __init__(self, net: DenseNetwork, update_rule: UpdateRule, loss_function: LossFunction):
        self.__net = net
        self.__update_rule = update_rule
        self.__grad = None

    def reset_grad(self):
        pass

    def optimize(self, x: numpy.array, t: numpy.array) -> numpy.array:
        pass

    def update(self):
        pass

    def __back_prop(self):
        (a, y) = self.__net(x)

