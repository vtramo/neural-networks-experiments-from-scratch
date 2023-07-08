from nnkit.datasets import mnist
from nnkit.datasets.utils import DataLabelSet, one_hot
from nnkit.training.model_selection import KFold
from nnkit.training.neurotrain import NetworkTrainer
from nnkit.training.update_rules import IRPropPlus
from nnkit.training.metrics import Accuracy
from nnkit.core.neuronet import DenseLayer, DenseNetwork
from nnkit.core.activations import Softmax, ReLU
from nnkit.core.losses import CrossEntropySoftmax
from nnkit.plotlib import save_histories_to_file


if __name__ == '__main__':

    net_32ReLU = DenseNetwork(
        DenseLayer(num_inputs=784, num_neurons=32, activation_function=ReLU()),
        DenseLayer(num_neurons=10, activation_function=Softmax())
    )

    net_512ReLU = DenseNetwork(
        DenseLayer(num_inputs=784, num_neurons=512, activation_function=ReLU()),
        DenseLayer(num_neurons=10, activation_function=Softmax())
    )

    # Load data / Data pre-processing
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = (train_images.astype('float32') / 255)[:5000]
    train_labels = one_hot(train_labels)[:5000]
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    test_labels = one_hot(test_labels)

    # Evaluate different architectures
    training_set = DataLabelSet(train_images, train_labels, batch_size=len(train_images), name='training')

    networks = [
        ('net_32ReLU', net_32ReLU),
        ('net_512ReLU', net_512ReLU)
    ]

    kfold = KFold()
    for net_name, net in networks:
        histories = []
        for i, (train_set, test_set) in enumerate(kfold(training_set)):
            trainer = NetworkTrainer(
                net=net,
                update_rule=IRPropPlus(),
                loss_function=CrossEntropySoftmax(),
                metrics=[Accuracy()]
            )

            history = trainer.train_network(training_set=train_set, validation_set=test_set, epochs=10)
            histories.append(history)
            trainer.net.reset_parameters()

            best_parameters = history.best_parameters
            best_test_accuracy = best_parameters.metric_results.metric_by_name['test_accuracy']
            print(f'KFold iteration {i} - {net_name} - best_test_accuracy: {test_accuracy.result()}')

        path = f'{net_name}-train-histories-kfold.pkl'
        save_histories_to_file(histories, path=path)
