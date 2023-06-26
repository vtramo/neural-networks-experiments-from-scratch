import nnkit
from nnkit.neuronet import DenseLayer, DenseNetwork
from nnkit.actfun import Sigmoid, Softmax, ReLU
from nnkit.lossfun import CrossEntropySoftmax
from nnkit.datasets import mnist
from nnkit.neurotrain import DataLabelSet, DataLabelBatchGenerator, NetworkTrainer, SGD, Accuracy, RProp


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
    train_labels_hot = nnkit.one_hot(train_labels)[:10000]

    validation_split = 0.2
    validation_set_size = int(len(train_images) * validation_split)
    validation_images = train_images[-validation_set_size:]
    validation_labels = train_labels_hot[-validation_set_size:]

    loss = CrossEntropySoftmax()
    update_rule = RProp(learning_rate=0.01)
    trainer = NetworkTrainer(net=net, update_rule=update_rule, loss=loss, metrics=[Accuracy()])
    training_set = DataLabelBatchGenerator(train_images, train_labels_hot, batch_size=128)
    validation_set = DataLabelSet(validation_images, validation_labels)
    trainer.train_network(training_set, validation_set, epochs=5)
