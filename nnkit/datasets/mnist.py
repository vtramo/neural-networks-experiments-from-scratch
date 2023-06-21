import gzip
import numpy as np


def _load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return np.reshape(data, (-1, 28, 28))


def _load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


def load_data():
    train_images = _load_mnist_images('nnkit/datasets/mnist/train-images-idx3-ubyte.gz')
    train_labels = _load_mnist_labels('nnkit/datasets/mnist/train-labels-idx1-ubyte.gz')
    test_images = _load_mnist_images('nnkit/datasets/mnist/t10k-images-idx3-ubyte.gz')
    test_labels = _load_mnist_labels('nnkit/datasets/mnist/t10k-labels-idx1-ubyte.gz')

    train_images_reshaped = np.reshape(train_images, (-1, 28, 28))
    test_images_reshaped = np.reshape(test_images, (-1, 28, 28))

    return (train_images_reshaped, train_labels), (test_images_reshaped, test_labels)
