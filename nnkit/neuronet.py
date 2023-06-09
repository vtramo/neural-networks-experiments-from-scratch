from .actfun import DerivableFunction, IdentityFunction
import numpy


class Neuron:
    """
    A class representing a single neuron in a neural network.

    Attributes:
        activation_function: The activation function used by the neuron.

    Methods:
        __call__(x, w) -> float: Compute the output of the neuron given the input and weights.
    """

    def __init__(self, activation_function: DerivableFunction = IdentityFunction()):
        self.__activation_function = activation_function

    @property
    def activation_function(self) -> DerivableFunction:
        return self.__activation_function

    def __call__(self, x: numpy.array, w: numpy.array) -> float:
        a = numpy.dot(x, w)
        return self.__activation_function(a)


class DenseLayer:
    """
    A class representing a dense layer in a neural network.

    Attributes:
        activation_function: The activation function used by the layer.
        neurons: The neurons in the layer.

    Methods:
        __call__(inputs: numpy.array) -> numpy.array:
            Compute the output of the layer given the inputs (or the output of the previous layer).

    """

    def __init__(self, num_neurons: int, activation_function: DerivableFunction = IdentityFunction()):
        self.__activation_function = activation_function
        self.__weights = None
        self.__create_neurons(num_neurons)

    def __create_neurons(self, num_neurons: int) -> None:
        self.__neurons = [
            Neuron(self.__activation_function)
            for _ in range(0, num_neurons)
        ]

    def __call__(self, inputs: numpy.array) -> numpy.array:
        if not self.__is_weights_initialized():
            self.__initialize_weights(len_inputs=len(inputs))

        return self.__forward_pass(inputs)

    def __is_weights_initialized(self) -> bool:
        return self.__weights is not None

    def __initialize_weights(self, len_inputs: int) -> None:
        num_neurons = len(self.__neurons)
        self.__weights = numpy.random.rand(num_neurons, len_inputs + 1)  # bias included

    def __forward_pass(self, inputs: numpy.array) -> numpy.array:
        inputs = numpy.concatenate(([1], inputs))  # include 1 for bias
        return numpy.array([
            neuron(inputs, self.__weights[i])
            for i, neuron in enumerate(self.__neurons)
        ])
