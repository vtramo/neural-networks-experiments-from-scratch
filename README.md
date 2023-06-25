# Neural Networks Experiments From Scratch

This repository contains code for creating and training neural networks using only basic libraries such as [NumPy](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi-u9-1r7T_AhXT8rsIHaNlAdoQFnoECAwQAQ&url=https%3A%2F%2Fnumpy.org%2F&usg=AOvVaw3L2i9HVc9ZeynETpNrPxO-) for array manipulation and numerical calculations.

## Usage example

```python
import nnkit
from nnkit.neuronet import DenseLayer, DenseNetwork
from nnkit.actfun import Sigmoid, Softmax, ReLU
from nnkit.lossfun import CrossEntropySoftmax
from nnkit.datasets import mnist
from nnkit.neurotrain import DataLabelBatchGenerator, NeuralNetworkSupervisedTrainer, SGD

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

loss = CrossEntropySoftmax()
update_rule = SGD(learning_rate=0.1)
trainer = NeuralNetworkSupervisedTrainer(net=net, update_rule=update_rule, loss=loss)
dataset_generator = DataLabelBatchGenerator(train_images, train_labels_hot, batch_size=128)
trainer.train_network(data_label_batch_generator=dataset_generator, epochs=5)
```
