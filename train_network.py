from nnkit.core.neuronet import DenseLayer, DenseNetwork
from nnkit.core.activations import Softmax, ReLU, Sigmoid
from nnkit.core.losses import CrossEntropySoftmax
from nnkit.datasets import mnist
from nnkit.datasets.utils import DataLabelSet, one_hot
from nnkit.training.neurotrain import NetworkTrainer
from nnkit.training.update_rules import SGD, RPropPlus, IRPropPlus, RPropMinus, IRPropMinus
from nnkit.training.stopping import GLStoppingCriterion
from nnkit.training.metrics import Accuracy

if __name__ == '__main__':
    # Build Network
    net = DenseNetwork(
        DenseLayer(num_inputs=784, num_neurons=64, activation_function=Sigmoid()),
        DenseLayer(num_neurons=32, activation_function=Sigmoid()),
        DenseLayer(num_neurons=10, activation_function=Softmax())
    )

    # Load data / Data pre-processing
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))[:200]
    train_images = (train_images.astype('float32') / 255)
    train_labels = one_hot(train_labels)[:200]
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    test_labels = one_hot(test_labels)

    # Training data / Validation data
    training_set = DataLabelSet(train_images, train_labels, batch_size=len(train_images), name='training')
    training_set, validation_set = training_set.split(
        split_factor=0.2,
        split_set_batch_size=len(train_images),
        split_set_name='validation'
    )

    # Train the network
    trainer = NetworkTrainer(
        net=net,
        update_rule=RPropPlus(),
        loss_function=CrossEntropySoftmax(),
        metrics=[Accuracy()]
    )

    history = trainer.train_network(training_set, validation_set, epochs=3)
