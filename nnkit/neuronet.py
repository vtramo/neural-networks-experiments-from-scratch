from .actfun import ActivationFunction, Identity, Softmax
import numpy as np


class Neuron:

    def __init__(self, activation_function: ActivationFunction = Identity()):
        self.__activation_function = activation_function

    @property
    def activation_function(self) -> ActivationFunction:
        return self.__activation_function

    def __call__(self, x: np.ndarray, w: np.ndarray) -> float:
        a = np.dot(x, w)
        z = self.__activation_function(a)
        return z
    
    def training_output(self, x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
        a = np.dot(x, w)
        z = self.__activation_function(a)
        d = self.__activation_function.derivative(a)
        return z, d


class DenseLayer:

    def __init__(
        self,
        num_neurons: int,
        num_inputs: int = None,
        activation_function: ActivationFunction = Identity()
    ):
        is_softmax = isinstance(activation_function, Softmax)
        self.__activation_function = Identity() if is_softmax else activation_function
        self.__post_processing = activation_function if is_softmax else Identity()
        self.__neurons = [Neuron(self.__activation_function) for _ in range(0, num_neurons)]
        self.__weights = self.initialize_weights(num_inputs) if num_inputs is not None else None

    def initialize_weights(self, num_inputs: int) -> np.ndarray:
        self.__weights = np.random.rand(self.num_neurons, num_inputs + 1)
        return self.__weights

    @property
    def weights(self) -> np.ndarray:
        if not self.are_weights_initialised():
            raise Exception("Weights are not initialized!")

        return self.__weights

    @weights.setter
    def weights(self, weights: np.ndarray):
        self.__weights = weights

    def are_weights_initialised(self) -> bool:
        return self.__weights is not None

    @property
    def num_neurons(self) -> int:
        return len(self.__neurons)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if not self.are_weights_initialised():
            self.initialize_weights(num_inputs=len(inputs))

        return self.__forward_pass(inputs)

    def __forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.concatenate(([1], inputs))

        output = np.array([
            neuron(inputs, self.weights[i])
            for i, neuron in enumerate(self.__neurons)
        ])

        return self.__post_processing(output)

    def training_forward_pass(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        inputs = np.concatenate(([1], inputs))

        z = np.zeros(self.num_neurons)
        d = np.zeros(self.num_neurons)
        for i, neuron in enumerate(self.__neurons):
            neuron_output = neuron.training_output(inputs, self.weights[i])
            z[i] = neuron_output[0]
            d[i] = neuron_output[1]

        return {'z': self.__post_processing(z), 'd': d}


class DenseNetwork:
    def __init__(self, *layers: DenseLayer):
        DenseNetwork.__check_dense_layers(*layers)
        self.__layers = layers
        self.__parameters = np.zeros(len(layers), dtype=object)
        self.__init_parameters(*layers)

    @staticmethod
    def __check_dense_layers(*layers: DenseLayer) -> None:
        if len(layers) <= 0:
            raise Exception("The number of DenseLayers cannot be less than or equal to zero")

        for layer in layers:
            if not isinstance(layer, DenseLayer):
                raise TypeError("layers must be DenseLayers")

        if not layers[0].are_weights_initialised():
            raise Exception("The first DenseLayer must be initialised!")

    def __init_parameters(self, *layers: DenseLayer) -> None:
        input_layer = layers[0]
        self.__parameters[0] = input_layer.weights

        prev_layer = input_layer
        for index_layer, layer in enumerate(layers[1:], start=1):
            layer_weights = layer.initialize_weights(prev_layer.num_neurons)
            self.__parameters[index_layer] = layer_weights
            prev_layer = layer

    @property
    def parameters(self) -> np.ndarray:
        return self.__parameters

    @parameters.setter
    def parameters(self, new_parameters: np.ndarray):
        self.__parameters = new_parameters
        for layer, new_params in zip(self.__layers, new_parameters):
            layer.weights = new_params

    @property
    def depth(self) -> int:
        return len(self.__layers)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.__output(x)

    def __output(self, x: np.ndarray) -> np.ndarray:
        output: np.ndarray = None

        prev_output = x
        for layer in self.__layers:
            output = layer(prev_output)
            prev_output = output

        return output

    def training_forward_pass(self, x: np.ndarray) -> tuple[dict[str, np.ndarray], ...]:
        output = []

        prev_output = x
        for layer in self.__layers:
            curr_output = layer.training_forward_pass(prev_output)
            output.append(curr_output)
            prev_output = curr_output['z']

        return tuple(output)
        