from nnkit.neuronet import DenseLayer, DenseNetwork
from nnkit.actfun import Softmax, ReLU
from nnkit.lossfun import CrossEntropySoftmax
from nnkit.datasets import mnist
from nnkit.datasets.utils import DataLabelSet, DataLabelBatchGenerator, one_hot
from nnkit.training.neurotrain import NetworkTrainer
from nnkit.training.update_rules import SGD, RPropPlus, IRPropPlus, RPropMinus, IRPropMinus
from nnkit.training.metrics import Accuracy


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

    training_data_size = 100
    training_split = 0.8
    validation_split = 0.2

    training_set_size = int(training_data_size * training_split)
    train_images = train_images[:training_set_size]
    train_labels_hot = one_hot(train_labels)[:training_set_size]

    validation_set_size = int(training_data_size * validation_split)
    validation_images = train_images[-validation_set_size:]
    validation_labels = train_labels_hot[-validation_set_size:]

    training_set = DataLabelBatchGenerator(train_images, train_labels_hot, batch_size=1)
    validation_set = DataLabelSet(validation_images, validation_labels)

    trainer = NetworkTrainer(
        net=net,
        update_rule=SGD(learning_rate=0.1, momentum=0.9),
        loss=CrossEntropySoftmax(),
        metrics=[Accuracy()]
    )

    trainer.train_network(training_set, validation_set, epochs=50)
