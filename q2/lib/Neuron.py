import numpy as np
from numpy.linalg import norm


class Neuron(object):
    def __init__(self, num_weights, beta, mu, output=0.0, leaning_rate=0.125):
        self.beta = 1.0
        self.mu = mu
        self.correct = [0] * num_weights
        self.output = output
        self.learning_rate = leaning_rate
        self.num_weights = num_weights
        self.reset_weights()

    def update_output(self, vector):
        # Execute gaussian activation function to update neuron output value
        self.output = gaussian(vector, self.beta, self.mu)

    def update_correction(self, error, index):
        self.correct[index] = error * self.output

    def apply_corrections(self):
        for i in range(0, len(self.weights)):
            self.weights[i] = self.weights[i] + (self.learning_rate * self.correct[i])

    def reset_weights(self):
        self.weights = np.random.uniform(-1, 1, self.num_weights)


def gaussian(x, beta, mu):
    """
    RBF Gaussian activation function.

    See http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
    """
    return np.exp(-beta * (norm(x - mu) ** 2.0))


def differentiated_gaussian(x, beta, mu):
    """
    Differentiated gaussian activation function computed by Wolfram Alpha.

    See https://www.wolframalpha.com/input/?i=d%2Fdx+e%5E(-beta*norm(x-mu)%5E2).
    """
    return -2 * beta * norm(x - mu) * gaussian(x, beta, mu)
