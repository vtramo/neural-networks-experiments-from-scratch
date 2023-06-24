import unittest
import numpy as np
from nnkit import DenseNetwork, DenseLayer, ReLU, Softmax, backprop, CrossEntropySoftmax


class TestBackprop(unittest.TestCase):

    def setUp(self) -> None:
        self.net = DenseNetwork(
            DenseLayer(num_inputs=4, num_neurons=2, activation_function=ReLU()),
            DenseLayer(num_neurons=3, activation_function=ReLU()),
            DenseLayer(num_neurons=2, activation_function=Softmax())
        )
        self.x = np.array([0.0, 0.33333334, 0.9882353, 0.9764706])
        self.t = np.array([0, 1])
        self.loss = CrossEntropySoftmax()
        self.net.parameters = self._static_parameters()

    @staticmethod
    def _static_parameters():
        parameters = np.zeros(3, dtype=object)
        parameters[0] = np.array([[-0.00146366, 0.0012152, 0.00029115, 0.00033854, 0.00299581],
                                  [-0.00069839, -0.00023209, 0.00101497, 0.00113509, -0.0004924]])
        parameters[1] = np.array([[1.07639644e-04, -5.74666749e-04, -8.89891805e-04],
                                  [-1.34914473e-03, 4.54462587e-05, 7.51359738e-04],
                                  [-6.74126629e-04, 1.05050734e-03, -6.59039883e-04]])
        parameters[2] = np.array([[-4.32547314e-04, -2.31855774e-03, -4.08447073e-04, 4.42932406e-04],
                                  [-8.67475798e-04, -2.07714950e-03, 9.46591126e-05, -1.69379848e-03]])
        return parameters

    def test_backprop(self):
        expected_gradient = [
            np.array([[6.93797277e-08, 0, 2.31265764e-08, 6.85634960e-08, 6.77472643e-08],
                      [1.07436964e-07, 0, 3.58123222e-08, 1.06173001e-07, 1.04909037e-07]]),
            np.array([[-1.20730367e-04, -2.28574889e-07, -3.39077543e-08],
                      [0, 0, 0],
                      [0, 0, 0]]),
            np.array([[5.00108726e-01, 5.31624154e-05, 0, 0],
                      [-5.00108726e-01, -5.31624154e-05, 0, 0]])
        ]

        gradient = backprop(self.net, self.loss, self.x, self.t)

        for gradient_layer, expected_gradient_layer in zip(gradient, expected_gradient):
            np.testing.assert_almost_equal(gradient_layer, expected_gradient_layer)
