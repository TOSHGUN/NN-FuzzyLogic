import numpy as np


def sigmoid(x):
    """
    Activation function f(x) = 1 / (1 + e^(-x))
    :param x:
    :return: f(x)
    """
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    """
    Производная от sigmoid: f'(x) = f(x) * (1 - f(x))
    :param x:
    :return:
    """
    fx = sigmoid(x)
    return fx * (1 - fx)


class Neuron:
    """
    Single neuron
    """
    def __init__(self, weights, bias):
        """

        :param weights:
        :param bias:
        """
        print("neuron init")
        self.weights = weights
        self.bias = bias

    def feed_forward(self, inputs):
        """

        :param inputs:
        :return:
        """
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
