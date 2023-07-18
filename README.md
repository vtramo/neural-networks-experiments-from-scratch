# Neural Networks Experiments From Scratch

This repository contains code for creating and training neural networks using only basic libraries such as [NumPy](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi-u9-1r7T_AhXT8rsIHaNlAdoQFnoECAwQAQ&url=https%3A%2F%2Fnumpy.org%2F&usg=AOvVaw3L2i9HVc9ZeynETpNrPxO-) for array manipulation and numerical calculations.

## Usage example

```python
from nnkit.core.neuronet import DenseLayer, DenseNetwork
from nnkit.core.activations import Softmax, ReLU, Sigmoid, Tanh
from nnkit.core.losses import CrossEntropySoftmax
from nnkit.datasets import mnist
from nnkit.datasets.utils import DataLabelSet, one_hot
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
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = (train_images.astype('float32') / 255)
    train_labels = one_hot(train_labels)
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
        update_rule=IRPropPlus(),
        loss_function=CrossEntropySoftmax(),
        metrics=[Accuracy()]
    )

    history = trainer.train_network(training_set, validation_set, epochs=30)

    # Test the network
    test_set = DataLabelSet(test_images, test_labels, batch_size=len(test_images), name='test')
    evaluator = MetricsEvaluator(net, metrics=[Accuracy()], loss_function=CrossEntropySoftmax())
    metrics = evaluator.compute_metrics(test_set)
    print(metrics)
```
