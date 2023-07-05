from matplotlib import pyplot as plt
import numpy as np
from nnkit.neuronet import DenseLayer, DenseNetwork
from nnkit.activations import Softmax, ReLU
from nnkit.losses import CrossEntropySoftmax
from nnkit.datasets import mnist
from nnkit.datasets.utils import DataLabelSet, one_hot
from nnkit.plotlib import plot_training_histories, plot_training_history
from nnkit.training.model_selection import KFold
from nnkit.training.neurotrain import NetworkTrainer, TrainingHistory
from nnkit.training.update_rules import SGD, RPropPlus, IRPropPlus, RPropMinus, IRPropMinus
from nnkit.training.metrics import Accuracy

if __name__ == '__main__':

    # Build Network
    net = DenseNetwork(
        DenseLayer(num_inputs=784, num_neurons=500, activation_function=ReLU()),
        DenseLayer(num_neurons=200, activation_function=ReLU()),
        DenseLayer(num_neurons=10, activation_function=Softmax())
    )

    # Load data / Data pre-processing
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = (train_images.astype('float32') / 255)[:100]
    train_labels = one_hot(train_labels)[:100]
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    test_labels = one_hot(test_labels)

    # Evaluate the network
    training_set = DataLabelSet(train_images, train_labels, name='training')
    training_set, validation_set = training_set.split(split_factor=0.3, split_set_name='validation')

    histories = []

    for update_rule in [RPropPlus(), IRPropPlus(), RPropMinus(), IRPropMinus()]:
        
        trainer = NetworkTrainer(
            net=net,
            update_rule=update_rule,
            loss_function=CrossEntropySoftmax(),
            metrics=[Accuracy(name='accuracy')]
        )

        trainer.net.reset_parameters()
        history = trainer.train_network(training_set, validation_set, epochs=5)

        histories.append(history)
    
    plot_training_histories(histories, 'validation_accuracy', show_plot=True, path='validation_accuracy.png')
