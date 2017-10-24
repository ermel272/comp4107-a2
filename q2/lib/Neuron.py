import numpy as np
from numpy.linalg import norm


class Neuron(object):
    def __init__(self, num_weights, beta, mu, output=0.0):
        self.beta = beta
        self.mu = mu

        self.bias = np.random.rand() - 0.5
        self.correct = None
        self.output = output
        self.weights = [i - 0.5 for i in np.random.rand(num_weights)]  # incoming weights from previous layer

    def update_output(self, vector):
        # Execute gaussian activation function to update neuron output value
        self.output = gaussian(vector, self.beta, self.mu)


def gaussian(x, beta, mu):
    """
    RBF Gaussian activation function.

    See http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
    """
    return np.e ** (-beta * (norm(x - mu) ** 2))
