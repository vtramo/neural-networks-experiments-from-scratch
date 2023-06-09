import numpy


class DerivableFunction:
    """
    An abstract base class representing a derivable function.

    Methods:
        __call__(x: numpy.array) -> numpy.array:
            Compute the function value for the given input.

        derivative(x: numpy.array) -> numpy.array:
            Compute the derivative of the function for the given input.

    """

    def __init__(self):
        pass

    def __call__(self, x: numpy.array) -> numpy.array:
        raise NotImplementedError

    def derivative(self, x: numpy.array) -> numpy.array:
        raise NotImplementedError


class SigmoidActivationFunction(DerivableFunction):
    """
    A class representing the Sigmoid activation function.

    Methods:
        __call__(x: numpy.array) -> numpy.array:
            Compute the Sigmoid function value for the given input.

        derivative(x: numpy.array) -> numpy.array:
            Compute the derivative of the Sigmoid function for the given input.

    """

    def __init__(self):
        super().__init__()

    def __call__(self, x: numpy.array) -> numpy.array:
        return 1 / (1 + numpy.exp(-x))

    def derivative(self, x: numpy.array) -> numpy.array:
        sigma_x = self(x)
        return sigma_x * (1 - sigma_x)


class IdentityFunction(DerivableFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, x: float) -> float:
        return x

    def derivative(self, x: float) -> float:
        return 1
