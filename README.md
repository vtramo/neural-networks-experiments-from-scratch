# Neural Networks Experiments From Scratch

This repository contains code for creating and training neural networks using only basic libraries such as [NumPy](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi-u9-1r7T_AhXT8rsIHaNlAdoQFnoECAwQAQ&url=https%3A%2F%2Fnumpy.org%2F&usg=AOvVaw3L2i9HVc9ZeynETpNrPxO-) for array manipulation and numerical calculations.

## Usage example

```python
from nnkit.core.neuronet import DenseLayer, DenseNetwork
from nnkit.core.activations import Softmax, ReLU
from nnkit.core.losses import CrossEntropySoftmax
from nnkit.datasets import mnist
from nnkit.datasets.utils import DataLabelSet, one_hot
from nnkit.training.neurotrain import NetworkTrainer
from nnkit.training.update_rules import SGD, RPropPlus, IRPropPlus, RPropMinus, IRPropMinus
from nnkit.training.metrics import Accuracy

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
    train_images = (train_images.astype('float32') / 255)
    train_labels = one_hot(train_labels)
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    test_labels = one_hot(test_labels)

    # Training data / Validation data
    training_set = DataLabelSet(train_images, train_labels, batch_size=1, name='training')
    training_set, validation_set = training_set.split(
        split_factor=0.3,
        split_set_batch_size=len(train_images),
        split_set_name='validation'
    )

    # Train the network
    trainer = NetworkTrainer(
        net=net,
        update_rule=SGD(learning_rate=0.1, momentum=0.9),
        loss_function=CrossEntropySoftmax(),
        metrics=[Accuracy()],
        multiprocessing=False
    )

    history = trainer.train_network(training_set, validation_set, epochs=50)
```
