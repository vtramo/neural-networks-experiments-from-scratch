import numpy


class DerivableFunction:

    def __init__(self):
        pass

    def __call__(self, x: numpy.array) -> numpy.array:
        raise NotImplementedError

    def derivative(self, x: numpy.array) -> numpy.array:
        raise NotImplementedError


class Sigmoid(DerivableFunction):

    def __init__(self):
        super().__init__()

    def __call__(self, x: float) -> float:
        return 1 / (1 + numpy.exp(-x))

    def derivative(self, x: float) -> float:
        sigma_x = self(x)
        return sigma_x * (1 - sigma_x)


class Identity(DerivableFunction):

    def __init__(self):
        super().__init__()

    def __call__(self, x: float) -> float:
        return x

    def derivative(self, x: float) -> float:
        return 1


class Softmax(DerivableFunction):

    def __init__(self):
        super().__init__()

    def __call__(self, y: numpy.array) -> numpy.array:
        exp_y = numpy.exp(y)
        exp_sum_y = numpy.sum(exp_y)
        return numpy.array([
            numpy.exp(y_value) / exp_sum_y
            for y_value in y
        ])

    def derivative(self, x: float) -> float:
        raise NotImplementedError
