import pprint
from nnkit import Sigmoid, Softmax, DenseLayer, DenseNetwork, backprop, CrossEntropySoftmax
import numpy as np
import gzip

if __name__ == '__main__':
    sigmoid = Sigmoid()
    softmax = Softmax()


    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)
        return data

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    train_images = load_mnist_images('mnist/train-images-idx3-ubyte.gz')
    train_labels = load_mnist_labels('mnist/train-labels-idx1-ubyte.gz')

    test_images = load_mnist_images('mnist/t10k-images-idx3-ubyte.gz')
    test_labels = load_mnist_labels('mnist/t10k-labels-idx1-ubyte.gz')

    net = DenseNetwork(
        DenseLayer(num_inputs=784, num_neurons=10, activation_function=sigmoid),
        DenseLayer(num_neurons=128, activation_function=sigmoid),
        DenseLayer(num_neurons=10, activation_function=softmax)
    )

    image = train_images[0]
    image_vector = image.reshape(-1)
    normalized_image = image_vector / 255.0
    net_output = net(normalized_image)

    loss = CrossEntropySoftmax()

    def calculate_accuracy(net, images, labels):
        correct_predictions = 0
        for image, label in zip(images, labels):
            image_vector = image.reshape(-1)
            normalized_image = image_vector / 255.0
            net_output = net(normalized_image)
            prediction = np.argmax(net_output)
            if prediction == label:
                correct_predictions += 1
        return correct_predictions / len(images)
    
    learning_rate = 0.1
    num_epochs = 10

    for epoch in range(num_epochs):
        total_gradient = [np.zeros_like(param) for param in net.parameters]
        
        for image, label in zip(train_images, train_labels):
            image_vector = image.reshape(-1)
            normalized_image = image_vector / 255.0
            net_output = net(normalized_image)
            
            gradient = backprop(net, loss, x=normalized_image, t=label)
            total_gradient = [total + grad for total, grad in zip(total_gradient, gradient)]

        new_parameters = [param - learning_rate * total_grad for param, total_grad in zip(net.parameters, total_gradient)]
        net.set_parameters(new_parameters)
        
        accuracy = calculate_accuracy(net, test_images, test_labels)
        
        print(f"Epoch {epoch}, Accuracy: {accuracy}")

