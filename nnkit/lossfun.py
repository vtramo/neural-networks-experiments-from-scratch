import numpy


class LossFunction:
    def __init__(self):
        pass

    def __call__(self, prediction: numpy.array, gold_label: numpy.array) -> float:
        raise NotImplementedError

    def output_derivative(self, prediction: numpy.array, gold_label: numpy.array) -> numpy.array:
        raise NotImplementedError


class CrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, prediction: numpy.array, gold_label: numpy.array) -> float:
        return -numpy.sum(prediction * numpy.log(gold_label))

    def output_derivative(self, prediction: numpy.array, gold_label: numpy.array) -> numpy.array:
        return -(gold_label / prediction) + ((1 - gold_label) / (1 - prediction))


class CrossEntropySoftmax(CrossEntropy):
    def output_derivative(self, prediction: numpy.array, gold_label: numpy.array) -> numpy.array:
        return prediction - gold_label


