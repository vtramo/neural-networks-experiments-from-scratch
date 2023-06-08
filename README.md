# Neural Networks Experiments From Scratch

This repository contains code for creating and training neural networks using only basic libraries such as [NumPy](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi-u9-1r7T_AhXT8rsIHaNlAdoQFnoECAwQAQ&url=https%3A%2F%2Fnumpy.org%2F&usg=AOvVaw3L2i9HVc9ZeynETpNrPxO-) for array manipulation and numerical calculations.

## Usage example

```python
from nnkit import Neuron, SigmoidActivationFunction, DenseLayer
import numpy as np

actfun = SigmoidActivationFunction()
neuron = Neuron(activation_function=actfun)

x = np.array([1, 2, 3])
w = np.array([1, 2, 3])
output = neuron(x, w)

layer = DenseLayer(10, activation_function=actfun)
layer_output = layer(x)
```
