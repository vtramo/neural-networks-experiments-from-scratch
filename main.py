from nnkit import Neuron, SigmoidActivationFunction, DenseLayer
import numpy as np


if __name__ == '__main__':
    actfun = SigmoidActivationFunction()
    neuron = Neuron(activation_function=actfun)

    x = np.array([1, 2, 3])
    w = np.array([1, 2, 3])
    output = neuron(x, w)
    print(f"Output neuron: {output}")
    print(f"Output derivative: {actfun.derivative(output)}")

    layer = DenseLayer(10, activation_function=actfun)
    layer_output = layer(x)
    print(f"\nlayer_output: \n{layer_output}")
