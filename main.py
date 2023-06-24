import copy
import math
from multiprocessing import Pool

import numpy.random
from matplotlib import pyplot as plt

from nnkit import Sigmoid, Softmax, DenseLayer, DenseNetwork, backprop, CrossEntropySoftmax, ReLU, UpdateRule, Optimizer, SGD
from nnkit.datasets import mnist
import numpy as np


def partition_into_chunks(iterable, chunk_size: int):
    return [
        iterable[i * chunk_size:(i * chunk_size) + chunk_size]
        for i in range(math.ceil(len(train_images) / chunk_size))
    ]


def calculate_accuracy_and_loss(net: DenseNetwork, images, labels, loss):
    correct_predictions = 0
    total_loss = 0

    for image, label in zip(images, labels):
        net_output = net(image)

        if np.argmax(net_output) == np.argmax(label):
            correct_predictions += 1

    return correct_predictions / len(images), total_loss / len(images)


def compute_gradient(net, loss, X, T) -> np.ndarray:
    gradient = np.zeros(net.parameters.shape, dtype=object)

    for x, t in zip(X, T):
        gradient += backprop(net, loss, x, t)

    return gradient

def _one_hot(labels, tot_classes=10):
    base = np.arange(tot_classes)
    return numpy.array([(base == label) * 1 for label in labels])


if __name__ == '__main__':

    net = DenseNetwork(
        DenseLayer(num_inputs=784, num_neurons=500, activation_function=ReLU()),
        DenseLayer(num_neurons=200, activation_function=ReLU()),
        DenseLayer(num_neurons=10, activation_function=Softmax())
    )

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255

    train_images = train_images[:10000]
    train_labels_hot = _one_hot(train_labels)[:10000]

    learning_rate = 0.5
    num_epochs = 1
    loss = CrossEntropySoftmax()
    processors = 4
    chunk_size = len(train_images) // processors

    train_images_chunks = partition_into_chunks(train_images, chunk_size)
    train_labels_hot_chunks = partition_into_chunks(train_labels_hot, chunk_size)

    opt=Optimizer(net,SGD(learning_rate=0.5),loss)

    for epoch in range(num_epochs):
        opt.reset_grad()

        with Pool(processors) as pool:
            backprop_arguments = [(net, loss, train_images_chunks[i], train_labels_hot_chunks[i]) for i in range(0, processors)]
            for result_gradient in pool.starmap(compute_gradient, backprop_arguments):
                opt.update_grad(result_gradient)

        opt.optimize()

        print(f"Epoch {epoch}")

    accuracy, loss_res = calculate_accuracy_and_loss(net, train_images, train_labels_hot, loss)
    print(f"Accuracy: {accuracy}, Loss: {loss_res}")