import random
from matplotlib import pyplot as plt
import numpy as np
from nnkit.neuronet import DenseLayer, DenseNetwork
from nnkit.activations import Softmax, ReLU
from nnkit.losses import CrossEntropySoftmax
from nnkit.datasets import mnist
from nnkit.datasets.utils import DataLabelSet, one_hot
from nnkit.plotlib import load_histories_from_file, plot_training_histories, plot_training_history, save_histories_to_file
from nnkit.training.model_selection import KFold
from nnkit.training.neurotrain import NetworkTrainer, TrainingHistory
from nnkit.training.update_rules import SGD, RPropPlus, IRPropPlus, RPropMinus, IRPropMinus
from nnkit.training.metrics import Accuracy, MetricsEvaluator
from datetime import datetime

def read_activation_functions(input: str):
    if input == 'relu':
        return ReLU()
    elif input == 'softmax':
        return Softmax()
    else:
        raise ValueError('Invalid activation function')
    
def save_config_to_file(config: dict, path: str):
    with open(path, 'w') as file:
        for key, value in config.items():
            file.write(str(key) + ': ' + str(value) + '\n')

def load_config_from_file(path: str):
    with open(path, 'r') as file:
        lines = file.readlines()
        config = {}
        for line in lines:
            key, value = line.split(':')
            config[key.strip()] = value
        return config
            

def get_data_from_input():

    print('Would you like to load a configuration from file? (y/n)')
    if input() == 'y':
        print('Enter the path to the configuration file:')
        config_path = input()
        config = load_config_from_file(f"configs/{config_path}")
        num_inputs = int(config['num_inputs'])
        num_layers = int(config['num_layers'])
        num_neurons = eval(config['num_neurons'])
        raw_activation_functions = eval(config['activation_functions'])
        activation_functions = [read_activation_functions(af) for af in raw_activation_functions]
        sample_size = int(config['sample_size'])
        epochs = int(config['epochs'])
        batch_size = int(config['batch_size'])
        split_factor = float(config['split_factor'])
        return num_inputs, num_layers, num_neurons, activation_functions, sample_size, epochs, batch_size, split_factor

    print('Enter the number of inputs:')
    num_inputs = int(input())
    print('Enter the number of layers: ')
    num_layers = int(input())
    print('Enter the number of neurons in each layer: ')
    num_neurons = [int(input()) for _ in range(num_layers)]
    print('Enter the activation function for each layer: (relu, softmax)')
    raw_activation_functions = [input() for _ in range(num_layers)]
    activation_functions = [read_activation_functions(af) for af in raw_activation_functions]
    print('Enter dataset sample size:')
    sample_size = int(input())
    print('Enter the number of epochs:')
    epochs = int(input())
    print('Enter the batch size:')
    batch_size = int(input())
    print('Enter validation split factor:')
    split_factor = float(input())

    print('Would you like to save this configuration? (y/n)')
    if input() == 'y':
        config = {
            'num_inputs': num_inputs,
            'num_layers': num_layers,
            'num_neurons': num_neurons,
            'activation_functions': raw_activation_functions,
            'sample_size': sample_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'split_factor': split_factor
        }
        save_config_to_file(config, f"configs/last_config")

    return num_inputs, num_layers, num_neurons, activation_functions, sample_size, epochs, batch_size, split_factor

def get_default_values():
    num_inputs = 784
    num_layers = 3
    num_neurons = [500, 200, 10]
    activation_functions = [ReLU(), ReLU(), Softmax()]
    sample_size = 60000
    epochs = 10
    batch_size = 60000
    split_factor = 0.3

    return num_inputs, num_layers, num_neurons, activation_functions, sample_size, epochs, batch_size, split_factor


if __name__ == '__main__':

    print('Want to use default values? [num_inputs=784, num_layers=3, num_neurons=(500,200,10), act_fun=(relu,relu,softmax), sample_size=60000, epochs=10, batch_size=60000, validation_split_factor=0.3] (y/n)')
    if input() == 'y':
        num_inputs, num_layers, num_neurons, activation_functions, sample_size, epochs, batch_size, split_factor = get_default_values()
    else:
        num_inputs, num_layers, num_neurons, activation_functions, sample_size, epochs, batch_size, split_factor = get_data_from_input()

    # Build Network
    net = DenseNetwork(
        DenseLayer(num_inputs=num_inputs, num_neurons=num_neurons[0], activation_function=activation_functions[0]),
        *[DenseLayer(num_neurons=num_neurons[i], activation_function=activation_functions[i]) for i in range(1, num_layers)]
    )

    # Load data / Data pre-processing
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = (train_images.astype('float32') / 255)[:sample_size]
    train_labels = one_hot(train_labels)[:sample_size]
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    test_labels = one_hot(test_labels)

    # Evaluate the network
    training_set = DataLabelSet(train_images, train_labels, name='training')
    training_set, validation_set = training_set.split(split_factor=0.3, split_set_name='validation')

    histories = []

    for update_rule in [RPropPlus(), IRPropPlus(), RPropMinus(), IRPropMinus()]:
        
        trainer = NetworkTrainer(
            net=net,
            update_rule=update_rule,
            loss_function=CrossEntropySoftmax(),
            metrics=[Accuracy(name='accuracy')]
        )

        trainer.net.reset_parameters()
        history = trainer.train_network(training_set, validation_set, epochs=epochs)

        histories.append(history)

    save_histories_to_file(histories, 'train_history.pkl')

    plot_training_histories(histories, 'validation_accuracy', show_plot=True, path='validation_accuracy.png')
