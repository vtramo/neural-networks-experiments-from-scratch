# Neural Networks Experiments From Scratch

This repository contains code for creating and training neural networks using only basic libraries such as [NumPy](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi-u9-1r7T_AhXT8rsIHaNlAdoQFnoECAwQAQ&url=https%3A%2F%2Fnumpy.org%2F&usg=AOvVaw3L2i9HVc9ZeynETpNrPxO-) for array manipulation and numerical calculations.

## Usage example

```python
from nnkit import Sigmoid, Softmax, DenseLayer, DenseNetwork
import numpy as np


sigmoid = Sigmoid()
softmax = Softmax()

net = DenseNetwork(
    DenseLayer(num_inputs=5, num_neurons=10, activation_function=sigmoid),
    DenseLayer(num_neurons=5, activation_function=sigmoid),
    DenseLayer(num_neurons=4, activation_function=softmax)
)

x = [1, 2, 3, 4, 5]
net_output = net(x)
print(f"{net_output}")
```
