from abc import ABCMeta, abstractmethod

import numpy

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
        self.__loss_function = loss_function
        self.__gradient = []

    def reset_grad(self):
        self.__gradient = []

    def optimize(self, x: numpy.array, t: numpy.array) -> numpy.array:
        pass

    def update(self):
        pass


def backprop(net: DenseNetwork, loss_function: LossFunction, x: numpy.array, t: numpy.array) -> numpy.array:
    net_output = net.training_forward_pass(x)
    net_parameters = net.parameters

    # Delta Output layer
    net_output_last_layer = net_output[-1]
    output_last_layer = net_output_last_layer['z']
    der_lossfun = loss_function.output_derivative(output_last_layer, t)
    der_actfun_last_layer = net_output_last_layer['d']
    delta_last_layer = der_lossfun * der_actfun_last_layer

    # Delta Hidden Layers
    delta = [delta_last_layer]
    for index_layer in reversed(range(0, net.depth - 1)):
        parameters_next_layer = net_parameters[index_layer + 1]

        # remove bias
        weights_next_layer = numpy.array([
            parameters_next_layer[i][1:]
            for i in range(0, len(parameters_next_layer))
        ])

        net_output_curr_layer = net_output[index_layer]
        der_actfun_curr_layer = net_output_curr_layer['d']
        delta_next_layer = delta[0]
        delta_curr_layer = der_actfun_curr_layer * numpy.matmul(weights_next_layer.transpose(), delta_next_layer)
        delta.insert(0, delta_curr_layer)

    # Compute gradient
    gradient = []
    for index_layer in reversed(range(0, net.depth)):
        delta_curr_layer = delta[index_layer]
        output_prev_layer = net_output[index_layer - 1]['z'] if index_layer != 0 else x
        der = [
            numpy.concatenate(([delta], delta * output_prev_layer))
            for delta in delta_curr_layer
        ]
        gradient.insert(0, der)

    return gradient
