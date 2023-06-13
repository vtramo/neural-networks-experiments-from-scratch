from .actfun import DerivableFunction, Identity, Softmax
import numpy


class Neuron:
    """
    A class representing a single neuron in a neural network.

    Attributes:
        activation_function: The activation function used by the neuron.

    Methods:
        __call__(x, w) -> float: Compute the output of the neuron given the input and weights.
    """

    def __init__(self, activation_function: DerivableFunction = Identity()):
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

    Methods:
        weights: Property that returns the weights of the layer.
        num_neurons: Property that returns the number of neurons in the layer.
        __call__(inputs: numpy.array) -> numpy.array: Performs a forward pass through the layer.
        are_weights_initialised() -> bool: Checks if the weights are initialized.
        initialize_weights(num_inputs: int, include_bias: bool = True) -> numpy.array: Initializes the weights of the layer.
        __forward_pass(inputs: numpy.array) -> numpy.array: Performs a forward pass through the neurons in the layer.
    """

    def __init__(
        self,
        num_neurons: int,
        num_inputs: int = None,
        activation_function: DerivableFunction = Identity(),
        include_bias: bool = True
    ):
        is_softmax = isinstance(activation_function, Softmax)
        self.__activation_function = Identity() if is_softmax else activation_function
        self.__post_processing = activation_function if is_softmax else Identity()
        self.__neurons = [
            Neuron(self.__activation_function)
            for _ in range(0, num_neurons)
        ]
        self.__weights = self.initialize_weights(num_inputs, include_bias) if num_inputs is not None else None
        self.__include_bias = include_bias

    @property
    def weights(self) -> numpy.array:
        if not self.are_weights_initialised():
            raise Exception("Weights are not initialized!")
        return self.__weights

    @property
    def num_neurons(self) -> int:
        return len(self.__neurons)

    def __call__(self, inputs: numpy.array) -> numpy.array:
        if not self.are_weights_initialised():
            self.initialize_weights(num_inputs=len(inputs))

        return self.__forward_pass(inputs)

    def are_weights_initialised(self) -> bool:
        return self.__weights is not None

    def initialize_weights(self, num_inputs: int, include_bias: bool = True) -> numpy.array:
        num_neurons = self.num_neurons

        self.__weights = numpy.random.rand(
            num_neurons,
            num_inputs + include_bias
        )

        return self.__weights

    def __forward_pass(self, inputs: numpy.array) -> numpy.array:
        if self.__include_bias:
            inputs = numpy.concatenate(([1], inputs))

        output = numpy.array([
            neuron(inputs, self.__weights[i])
            for i, neuron in enumerate(self.__neurons)
        ])

        return self.__post_processing(output)


class DenseNetwork:
    """
    A class representing a dense neural network composed of DenseLayer objects.

    Methods:
        __init_parameters(*layers: DenseLayer) -> None: Initializes the parameters of the network.
        parameters: Property that returns the parameters of the network.
        __call__(x: numpy.array) -> numpy.array: Performs a forward pass through the network.
        __forward_pass(x: numpy.array) -> numpy.array: Performs a forward pass through the layers of the network.
        __check_dense_layers(*layers: DenseLayer) -> None: Checks if the provided layers are of type DenseLayer.

    """
    
    def __init__(self, *layers: DenseLayer):
        DenseNetwork.__check_dense_layers(*layers)
        self.__layers = layers
        self.__init_parameters(*layers)

    def __init_parameters(self, *layers: DenseLayer) -> None:
        self.__parameters = []

        input_layer = layers[0]
        input_layer_weights = input_layer.weights
        self.__parameters.append(input_layer_weights)

        prev_layer = input_layer
        for layer in layers[1:]:
            layer_weights = layer.initialize_weights(prev_layer.num_neurons)
            self.__parameters.append(layer_weights)
            prev_layer = layer

    @property
    def parameters(self) -> list:
        return self.__parameters

    def __call__(self, x: numpy.array) -> numpy.array:
        return self.__forward_pass(x)

    def __forward_pass(self, x: numpy.array) -> numpy.array:
        output = x

        prev_output = output
        for layer in self.__layers:
            output = layer(prev_output)
            prev_output = output

        return output

    @staticmethod
    def __check_dense_layers(*layers: DenseLayer) -> None:
        if len(layers) <= 0:
            raise Exception("The number of DenseLayers cannot be less than or equal to zero")

        for layer in layers:
            if not isinstance(layer, DenseLayer):
                raise TypeError("layers must be DenseLayers")

        if not layers[0].are_weights_initialised():
            raise Exception("The first DenseLayer must be initialised!")
        