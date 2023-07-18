

import numpy as np
from nnkit.core.activations import Softmax, Tanh
from nnkit.core.losses import CrossEntropySoftmax
from nnkit.core.neuronet import DenseLayer, DenseNetwork
from nnkit.datasets import mnist
from nnkit.datasets.utils import DataLabelSet, one_hot
from nnkit.plotlib import load_parameters_from_file
from nnkit.training.metrics import Accuracy, MetricsEvaluator

if __name__ == '__main__':
    params = load_parameters_from_file('sgdmomentum_125000_online_256_10_tanh_softmax_40epochs.npy')

    net = DenseNetwork(
        DenseLayer(num_inputs=784, num_neurons=256, activation_function=Tanh()),
        DenseLayer(num_neurons=10, activation_function=Softmax())
    )

    net.parameters = params

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    test_labels = one_hot(test_labels)
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = (train_images.astype('float32') / 255)
    train_labels = one_hot(train_labels)

    # Evaluate the network
    test_set = DataLabelSet(test_images, test_labels, batch_size=len(test_images), name='test')
    train_set = DataLabelSet(train_images, train_labels, batch_size=len(train_images), name='train')

    evaluator = MetricsEvaluator(net, loss_function=CrossEntropySoftmax(), metrics=[Accuracy()])
    metrics = evaluator.compute_metrics(test_set)
    print(metrics)

