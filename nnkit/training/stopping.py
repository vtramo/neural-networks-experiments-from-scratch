from nnkit.training.metrics import Metrics
from nnkit.training.neurotrain import TrainingData

from abc import ABCMeta, abstractmethod
from collections.abc import Callable


class StoppingCriterion(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, training_data: TrainingData) -> bool:
        pass


def generalization_loss(val_loss: float, opt_val_loss: float) -> float:
    return 100 * ((val_loss / opt_val_loss) - 1)


class GLStoppingCriterion(StoppingCriterion):

    def __init__(self, alpha: float):
        self.__alpha = alpha
        self.__last_val_loss = None
        self.__opt_val_loss = float('inf')

    def __call__(self, training_data: TrainingData) -> bool:
        self.__last_val_loss = training_data.val_loss
        self.__opt_val_loss = min(self.__opt_val_loss, self.__last_val_loss)
        return generalization_loss(self.__last_val_loss, self.__opt_val_loss) > self.__alpha
