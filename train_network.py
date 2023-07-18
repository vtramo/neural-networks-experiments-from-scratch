from datetime import datetime
from nnkit.core.neuronet import DenseLayer, DenseNetwork
from nnkit.core.activations import Softmax, ReLU, Sigmoid, Tanh
from nnkit.core.losses import CrossEntropySoftmax
from nnkit.datasets import mnist
from nnkit.datasets.utils import DataLabelSet, one_hot
from nnkit.plotlib import save_histories_to_file, save_parameters_to_file
from nnkit.training.neurotrain import NetworkTrainer
from nnkit.training.update_rules import SGD, RPropPlus, IRPropPlus, RPropMinus, IRPropMinus
from nnkit.training.stopping import GLStoppingCriterion
from nnkit.training.metrics import Accuracy, MetricsEvaluator

if __name__ == '__main__':
    # Build Network
    net = DenseNetwork(
        DenseLayer(num_inputs=784, num_neurons=256, activation_function=Tanh()),
        DenseLayer(num_neurons=10, activation_function=Softmax())
    )

    # Load data / Data pre-processing
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))[:12500]
    train_images = (train_images.astype('float32') / 255)
    train_labels = one_hot(train_labels)[:12500]
    test_images = test_images.reshape((10000, 28 * 28))[:5000]
    test_images = test_images.astype('float32') / 255
    test_labels = one_hot(test_labels)[:5000]

    # Training data / Validation data
    training_set = DataLabelSet(train_images, train_labels, batch_size=1, name='training')
    training_set, validation_set = training_set.split(
        split_factor=0.2,
        split_set_batch_size=len(train_images),
        split_set_name='validation'
    )

    # Train the network
    trainer = NetworkTrainer(
        net=net,
        update_rule=SGD(learning_rate=0.1, momentum=0.9),
        loss_function=CrossEntropySoftmax(),
        metrics=[Accuracy()]
    )

    history = trainer.train_network(training_set, validation_set, epochs=40)

    # Evaluate the network
    test_set = DataLabelSet(test_images, test_labels, batch_size=len(test_images), name='test')

    evaluator = MetricsEvaluator(net, loss_function=CrossEntropySoftmax(), metrics=[Accuracy()])
    metrics = evaluator.compute_metrics(test_set)
    print(metrics)

    datetime_today = f'{datetime.today()}'.replace(' ', '-').replace(':','-')
    histories_path = f'train-histories-{datetime_today}.pkl'
    save_histories_to_file([history], path=histories_path)
    save_parameters_to_file(net.parameters, path=f'parameters-{datetime_today}.pkl')

    

