import numpy as np
class Cell(object):
    def __init__(self, bias = 0.0, output = 0.0):
        self.output = output
        self.weights = []
    def set_output(self, output):
        self.output = output
    def init_weights(self, num_weights):
        invert_bias = np.random.rand() >= 0.5
        self.bias = (-1 if invert_bias else 1) * np.random.rand()
        self.weights = np.random.rand(num_weights)

        for i in range(len(self.weights)):
            if i % 2 is 0:
                self.weights[i] *= -1
