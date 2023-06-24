from .neuronet import DenseNetwork
from .lossfun import LossFunction
from abc import ABCMeta, abstractmethod
import numpy as np


class UpdateRule(object, metaclass=ABCMeta):

    def __init__(self, learning_rate: float = 0.1):
        self._learning_rate = learning_rate

    @abstractmethod
    def __call__(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        pass


class SGD(UpdateRule):
    
    def __call__(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return parameters - self._learning_rate * gradient


class Optimizer:

    def __init__(self, net: DenseNetwork, update_rule: UpdateRule, loss_function: LossFunction):
        self.__net = net
        self.__update_rule = update_rule
        self.__loss_function = loss_function
        self.__gradient = np.zeros(net.parameters.shape, dtype=object)

    def reset_grad(self):
        self.__gradient = np.zeros(self.__net.parameters.shape, dtype=object)

    def optimize(self):
        self.__net.parameters = self.__update_rule(self.__net.parameters, self.__gradient)
        
    def update_grad(self, gradient: np.ndarray):
        self.__gradient += gradient

