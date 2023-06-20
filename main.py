import pprint
from nnkit import Sigmoid, Softmax, DenseLayer, DenseNetwork, backprop, CrossEntropySoftmax
import numpy as np


if __name__ == '__main__':
    sigmoid = Sigmoid()
    softmax = Softmax()

    net = DenseNetwork(
        DenseLayer(num_inputs=5, num_neurons=10, activation_function=sigmoid),
        DenseLayer(num_neurons=5, activation_function=sigmoid),
        DenseLayer(num_neurons=4, activation_function=softmax)
    )

    x = np.array([1, 2, 3, 4, 5])
    net_output = net(x)
    net_forward_pass = net.training_forward_pass(x)

    loss = CrossEntropySoftmax()
    gradient = backprop(net, loss, x, [1, 2, 3, 4])
    print(f"{gradient}")
