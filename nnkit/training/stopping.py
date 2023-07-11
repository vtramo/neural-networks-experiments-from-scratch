from nnkit.training.metrics import Metrics
from nnkit.training.neurotrain import TrainingData

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from queue import Queue


class StoppingCriterion(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, training_data: TrainingData) -> bool:
        pass


def generalization_loss(val_loss: float, opt_val_loss: float) -> float:
    return 100 * ((val_loss / opt_val_loss) - 1)


class GLStoppingCriterion(StoppingCriterion):

    def __init__(self, alpha: float):
        self._alpha = alpha
        self._last_val_loss = None
        self._opt_val_loss = float('inf')

    def __call__(self, training_data: TrainingData) -> bool:
        self._update_val_loss(training_data.val_loss)
        return generalization_loss(self._last_val_loss, self._opt_val_loss) > self._alpha

    def _update_val_loss(self, val_loss: float) -> None:
        self._last_val_loss = val_loss
        self._opt_val_loss = min(self._opt_val_loss, self._last_val_loss)


class PQStoppingCriterion(GLStoppingCriterion):

    def __init__(self, alpha: float, strip_size: int = 5):
        super().__init__(alpha)
        self._strip_size = strip_size
        self._strip_index = 0
        self._train_strip = []

    def __call__(self, training_data: TrainingData) -> bool:
        self._update_val_loss(training_data.val_loss)
        self._update_strip(training_data.train_loss)

        if self._strip_index < self._strip_size:
            return False

        self._reset_strip()
        train_progress = self._compute_training_progress()
        return generalization_loss(self._last_val_loss, self._opt_val_loss) / train_progress > self._alpha

    def _update_strip(self, train_loss: float) -> None:
        self._train_strip.append(train_loss)
        self._strip_index += 1

    def _compute_training_progress(self) -> float:
        return 1000 * ((sum(self._train_strip) / k * min(self._train_strip)) - 1)

    def _reset_strip(self) -> None:
        self._strip_index = 0
        self._train_strip = []


class UPStoppingCriterion(StoppingCriterion):

    def __init__(self, tot_strips: int, strip_size: int = 5):
        self._tot_strips = tot_strips
        self._strip_size = strip_size
        self._val_strips_queue = Queue()
        self._last_val_strip = []

    def __call__(self, training_data: TrainingData) -> bool:
        self._last_val_strip.append(training_data.val_loss)

        if len(self._last_val_strip) == self._strip_size:
            self._update_val_strips()

        if not self._tot_strips_reached():
            return False

        return self._check_val_strips()

    def _update_val_strips(self) -> None:
        self._val_strips_queue.put(self._last_val_strip)
        if self._tot_strips_reached():
            self._val_strips_queue.get_nowait()

    def _tot_strips_reached(self) -> bool:
        return self._val_strips_queue.qsize() == self._tot_strips

    def _check_val_strips(self) -> bool:
        for val_strip in reversed(self._val_strips_queue.queue):
            if val_strip[self._strip_size - 1] <= val_strip[0]:
                return False
        return True
