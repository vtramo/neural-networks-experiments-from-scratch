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


class SigmoidLogisticFunction(DerivableFunction):
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

    def __call__(self, x: float) -> float:
        return 1 / (1 + numpy.exp(-x))

    def derivative(self, x: float) -> float:
        sigma_x = self(x)
        return sigma_x * (1 - sigma_x)


class Identity(DerivableFunction):
    """
    A class representing the identity function, which returns the input value unchanged.

    Inherits from the DerivableFunction class.

    Methods:
        __init__(): Initializes the IdentityFunction object.
        __call__(x: float) -> float: Returns the input value unchanged.
        derivative(x: float) -> float: Returns the derivative of the identity function, which is always 1.

    """

    def __init__(self):
        super().__init__()

    def __call__(self, x: float) -> float:
        return x

    def derivative(self, x: float) -> float:
        return 1
