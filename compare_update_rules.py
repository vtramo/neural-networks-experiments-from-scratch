from nnkit.core.losses import CrossEntropySoftmax
from nnkit.training.metrics import Accuracy
from nnkit.training.update_rules import RPropPlus, IRPropPlus
from nnkit.datasets import mnist
from nnkit.datasets.utils import DataLabelSet, one_hot
from nnkit.training.neurotrain import NetworkTrainer
from nnkit.confignet import interactive_build_network
from nnkit.plotlib import (
    plot_training_histories,
    save_histories_to_file
)

from datetime import datetime


if __name__ == '__main__':

    # Build network from (command line/config file/default values)
    net, sample_size, epochs, batch_size, split_factor = interactive_build_network()

    # Load data / Data pre-processing
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = (train_images.astype('float32') / 255)[:sample_size]
    train_labels = one_hot(train_labels)[:sample_size]
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    test_labels = one_hot(test_labels)

    # Evaluate and store the results of the metrics of different architectures
    training_set = DataLabelSet(train_images, train_labels, batch_size=batch_size, name='training')
    training_set, validation_set = training_set.split(split_factor=split_factor, split_set_name='validation')

    histories = []

    for update_rule in [RPropPlus(), IRPropPlus()]:
        trainer = NetworkTrainer(
            net=net,
            update_rule=update_rule,
            loss_function=CrossEntropySoftmax(),
            metrics=[Accuracy()]
        )
        history = trainer.train_network(training_set, validation_set, epochs=epochs)
        histories.append(history)
        trainer.net.reset_parameters()

    datetime_today = f'{datetime.today()}'.replace(' ', '-')
    histories_path = f'train-histories-{datetime_today}.pkl'
    save_histories_to_file(histories, path=histories_path)

    plot_path = f'validation-accuracy-{datetime_today}.png'
    plot_training_histories(histories, 'validation_accuracy', show_plot=True, path=plot_path)
