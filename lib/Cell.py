import numpy as np
class Cell(object):
    def __init__(self, num_inputs = 0, learning_rate = 0.5):
        self.learning_rate = learning_rate
        self.inputs = np.zeros(num_inputs)
        self.weights = np.random.rand(num_inputs)
        self.output = 0

    def set_inputs(self, inputs):
        self.inputs = [1 if i else 0 for i in inputs]
    def update_weights(self, error_delta):
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * self.inputs[i] * error_delta
    def set_output(self, output):
        ""
    def compute_and_update_output(self):
        for i in range(len(self.inputs)):
            self.output += self.inputs[i] * self.weights[i]
        self.output /= len(self.inputs)
        return self.output
