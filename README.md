# Neural Networks Experiments From Scratch

This repository contains code for creating and training neural networks using only basic libraries such as [NumPy](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi-u9-1r7T_AhXT8rsIHaNlAdoQFnoECAwQAQ&url=https%3A%2F%2Fnumpy.org%2F&usg=AOvVaw3L2i9HVc9ZeynETpNrPxO-) for array manipulation and numerical calculations.

## Usage example

```python
from nnkit.neuronet import DenseLayer, DenseNetwork
from nnkit.activations import Softmax, ReLU
from nnkit.losses import CrossEntropySoftmax
from nnkit.datasets import mnist
from nnkit.datasets.utils import DataLabelSet, DataLabelBatchGenerator, one_hot
from nnkit.training.neurotrain import NetworkTrainer
from nnkit.training.update_rules import SGD, RPropPlus, IRPropPlus, RPropMinus, IRPropMinus
from nnkit.training.metrics import Accuracy

import numpy as np


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
    train_images = train_images.astype("float32") / 255
    train_labels = one_hot(train_labels)
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255
    test_labels = one_hot(test_labels)

    # Training data / Validation data
    val_split = 0.3
    num_samples_val_set = int(len(train_images) * val_split)
    train_images = train_images[num_samples_val_set:]
    train_labels = train_labels[num_samples_val_set:]
    val_images = train_images[:num_samples_val_set]
    val_labels = train_labels[:num_samples_val_set]

    # Train the network
    training_set = DataLabelBatchGenerator(train_images, train_labels, batch_size=len(train_images), name="training")
    validation_set = DataLabelSet(val_images, val_labels, name="validation")

    trainer = NetworkTrainer(
        net=net,
        update_rule=RPropPlus(),
        loss_function=CrossEntropySoftmax(),
        metrics=[Accuracy()]
    )

    history = trainer.train_network(training_set, validation_set, epochs=1000)
```
