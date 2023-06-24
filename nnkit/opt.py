from .neuronet import DenseNetwork
from .lossfun import LossFunction
from abc import ABCMeta, abstractmethod
import numpy as np


class UpdateRule(object, metaclass=ABCMeta):

    def __init__(self, learning_rate: float = 0.01):
        self._learning_rate = learning_rate

    @abstractmethod
    def __call__(self) -> np.ndarray:
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


def backprop(net: DenseNetwork, loss_function: LossFunction, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    net_output = net.training_forward_pass(x)
    net_parameters = net.parameters

    # Delta Output layer
    net_output_last_layer = net_output[-1]
    output_last_layer = net_output_last_layer['z']
    der_lossfun = loss_function.output_derivative(output_last_layer, t)
    der_actfun_last_layer = net_output_last_layer['d']
    delta_last_layer = der_lossfun * der_actfun_last_layer

    # Delta Hidden Layers
    delta_layers = np.zeros(net.depth, dtype=object)
    delta_layers[net.depth - 1] = delta_last_layer
    for index_layer in reversed(range(0, net.depth - 1)):
        parameters_next_layer = net_parameters[index_layer + 1]

        # remove bias
        weights_next_layer = np.array([
            parameters_next_layer[i][1:]
            for i in range(0, len(parameters_next_layer))
        ])

        net_output_curr_layer = net_output[index_layer]
        der_actfun_curr_layer = net_output_curr_layer['d']
        delta_next_layer = delta_layers[index_layer + 1]
        delta_curr_layer = der_actfun_curr_layer * np.matmul(weights_next_layer.transpose(), delta_next_layer)
        delta_layers[index_layer] = delta_curr_layer

    # Compute gradient
    gradient = np.zeros(net.parameters.shape, dtype=object)
    for index_layer in reversed(range(0, net.depth)):
        delta_curr_layer = delta_layers[index_layer]
        output_prev_layer = net_output[index_layer - 1]['z'] if index_layer != 0 else x
        der = np.array([
            np.concatenate(([delta], delta * output_prev_layer))
            for delta in delta_curr_layer
        ])
        gradient[index_layer] = der

    return gradient
