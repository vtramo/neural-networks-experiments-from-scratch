from nnkit import DenseNetwork, DenseLayer
from nnkit.core.activations import ReLU, Softmax, Sigmoid

from datetime import date


def read_activation_functions(activation_function: str):
    if activation_function == 'relu':
        return ReLU()
    elif activation_function == 'softmax':
        return Softmax()
    elif activation_function == 'sigmoid':
        return Sigmoid()
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
        path = f'configs/config-{date.today()}'
        save_config_to_file(config, path=path)

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


INTERACTIVE_MESSAGE_BUILD_NET = '''
Want to use default values? [num_inputs=784, num_layers=3, num_neurons=(500,200,10), act_fun=(relu,relu,softmax), \
sample_size=60000, epochs=10, batch_size=60000, validation_split_factor=0.3] (y/n)\
'''


def interactive_build_network() -> tuple[DenseNetwork, int, int, int, float]:
    print(INTERACTIVE_MESSAGE_BUILD_NET)

    (num_inputs,
     num_layers,
     num_neurons,
     activation_functions,
     sample_size,
     epochs,
     batch_size,
     split_factor) = get_default_values() if input() == 'y' else get_data_from_input()

    net = DenseNetwork(
        DenseLayer(num_inputs=num_inputs, num_neurons=num_neurons[0], activation_function=activation_functions[0]),
        *[DenseLayer(num_neurons=num_neurons[i], activation_function=activation_functions[i]) for i in
          range(1, num_layers)]
    )

    return net, sample_size, epochs, batch_size, split_factor
