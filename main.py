from nnkit.neuronet import DenseLayer, DenseNetwork
from nnkit.activations import Softmax, ReLU
from nnkit.losses import CrossEntropySoftmax
from nnkit.datasets import mnist
from nnkit.datasets.utils import DataLabelSet, DataLabelBatchGenerator, one_hot
from nnkit.training.neurotrain import NetworkTrainer
from nnkit.training.update_rules import SGD, RPropPlus, IRPropPlus, RPropMinus, IRPropMinus
from nnkit.training.metrics import Accuracy

import numpy as np


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
    train_images = (train_images.astype('float32') / 255)[:500]
    train_labels = one_hot(train_labels)[:500]
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    test_labels = one_hot(test_labels)

    # Training data / Validation data
    training_set = DataLabelSet(train_images, train_labels, name='training')
    training_set, validation_set = training_set.split(split_factor=0.3, split_set_name='validation')
    training_set = DataLabelBatchGenerator.from_data_label_set(training_set, batch_size=len(training_set))

    # Train the network
    trainer = NetworkTrainer(
        net=net,
        update_rule=RPropMinus(),
        loss_function=CrossEntropySoftmax(),
        metrics=[Accuracy()],
        multiprocessing=False
    )

    history = trainer.train_network(training_set, validation_set, epochs=2000)
