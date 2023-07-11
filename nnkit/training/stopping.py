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
