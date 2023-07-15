from nnkit.core.activations import ActivationFunction, Identity, Softmax
from nnkit.core.losses import LossFunction
import numpy as np


class Neuron:

    def __init__(self, activation_function: ActivationFunction = Identity()):
        self.__activation_function = activation_function

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
        self.__weights = np.random.normal(size=(self.num_neurons, num_inputs + 1)) * 0.001
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
        first_layer = layers[0]
        self.__parameters[0] = first_layer.weights

        prev_layer = first_layer
        for index_layer, layer in enumerate(layers[1:], start=1):
            layer_weights = layer.initialize_weights(prev_layer.num_neurons)
            self.__parameters[index_layer] = layer_weights
            prev_layer = layer

    def reset_parameters(self) -> None:
        self.__init_parameters(*self.__layers)

    @property
    def parameters(self) -> np.ndarray:
        return self.__parameters

    @parameters.setter
    def parameters(self, parameters: np.ndarray):
        self.__parameters = parameters
        for layer, layer_weights in zip(self.__layers, parameters):
            layer.weights = layer_weights

    @property
    def depth(self) -> int:
        return len(self.__layers)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim == 2:
            return np.array([self.__output(x) for x in inputs])
        else:
            return self.__output(inputs)

    def __output(self, x: np.ndarray) -> np.ndarray:
        output = x

        for layer in self.__layers:
            output = layer(output)

        return output

    def training_forward_pass(self, x: np.ndarray) -> tuple[dict[str, np.ndarray]]:
        training_output = []

        prev_layer_output = x
        for layer in self.__layers:
            layer_training_output = layer.training_forward_pass(prev_layer_output)
            training_output.append(layer_training_output)
            prev_layer_output = layer_training_output['z']

        return tuple(training_output)

    def compute_gradients(self, loss, points, labels) -> np.ndarray:
        gradients = np.zeros(self.parameters.shape, dtype=object)

        for point, label in zip(points, labels):
            gradients += self.backprop_onecycle(loss, point, label)

        return gradients

    def backprop(self, loss_function: LossFunction, point: np.ndarray, label: np.ndarray) -> np.ndarray:
        net_output = self.training_forward_pass(point)

        # Delta Output layer
        net_output_last_layer = net_output[-1]
        output_last_layer = net_output_last_layer['z']
        der_actfun_last_layer = net_output_last_layer['d']
        der_lossfun = loss_function.output_derivative(output_last_layer, label)
        delta_last_layer = der_lossfun * der_actfun_last_layer

        # Delta Hidden Layers
        delta_layers = np.zeros(self.depth, dtype=object)
        delta_layers[self.depth - 1] = delta_last_layer
        for index_layer in reversed(range(0, self.depth - 1)):
            parameters_next_layer = self.parameters[index_layer + 1]

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

        # Compute gradients
        gradients = np.zeros(self.parameters.shape, dtype=object)
        for index_layer in reversed(range(0, self.depth)):
            delta_curr_layer = delta_layers[index_layer]
            output_prev_layer = net_output[index_layer - 1]['z'] if index_layer != 0 else point
            gradients_curr_layer = np.array([
                np.concatenate(([delta], delta * output_prev_layer))
                for delta in delta_curr_layer
            ])
            gradients[index_layer] = gradients_curr_layer

        return gradients

    def backprop_onecycle(self, loss_function: LossFunction, point: np.ndarray, label: np.ndarray) -> np.ndarray:
        net_output = self.training_forward_pass(point)

        # Delta Output layer
        net_output_last_layer = net_output[-1]
        output_last_layer = net_output_last_layer['z']
        der_actfun_last_layer = net_output_last_layer['d']
        der_lossfun = loss_function.output_derivative(output_last_layer, label)
        delta_last_layer = der_lossfun * der_actfun_last_layer

        # Compute gradients + Compute Delta Hidden Layers
        gradients = np.zeros(self.parameters.shape, dtype=object)
        delta_layers = np.zeros(self.depth, dtype=object)
        delta_layers[self.depth - 1] = delta_last_layer
        for index_layer in reversed(range(0, self.depth)):
            parameters_curr_layer = self.parameters[index_layer]

            # remove bias
            weights_curr_layer = np.array([
                parameters_curr_layer[i][1:]
                for i in range(0, len(parameters_curr_layer))
            ])

            output_prev_layer = net_output[index_layer - 1]['z'] if index_layer != 0 else point
            delta_curr_layer = np.tile(delta_layers[index_layer], (output_prev_layer.shape[0], 1))
            gradient_weights = output_prev_layer * delta_curr_layer.transpose()
            gradient_biases = delta_layers[index_layer]
            gradients[index_layer] = np.array([
                np.concatenate(([gradient_bias], neuron_gradient_weights))
                for neuron_gradient_weights, gradient_bias in zip(gradient_weights, gradient_biases)
            ])
            if index_layer != 0:
                der_actfun_prev_layer = net_output[index_layer - 1]['d']
                delta_layers[index_layer - 1] = der_actfun_prev_layer * np.matmul(weights_curr_layer.transpose(), delta_layers[index_layer])

        return gradients
