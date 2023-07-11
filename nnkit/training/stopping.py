from nnkit.training.metrics import Metrics
from nnkit.training.neurotrain import TrainingData

from abc import ABCMeta, abstractmethod
from collections.abc import Callable


class StoppingCriterion(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, training_data: TrainingData) -> bool:
        pass


