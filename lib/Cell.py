import numpy as np
class Cell(object):
    def __init__(self, bias = 0.0, output = 0.0):
        self.output = output
        self.weights = []
    def set_output(self, output):
        self.output = output
    def init_weights(self, num_weights):
        self.bias = 0.7 * np.random.rand()
        self.weights = 0.7 * np.random.rand(num_weights)
