from nnkit import Sigmoid, Softmax, DenseLayer, DenseNetwork
import numpy as np


if __name__ == '__main__':
    sigmoid = Sigmoid()
    softmax = Softmax()

    net = DenseNetwork(
        DenseLayer(num_inputs=5, num_neurons=10, activation_function=sigmoid),
        DenseLayer(num_neurons=5, activation_function=sigmoid),
        DenseLayer(num_neurons=4, activation_function=softmax)
    )

    x = [1, 2, 3, 4, 5]
    net_output = net(x)
    print(f"{net_output}")
