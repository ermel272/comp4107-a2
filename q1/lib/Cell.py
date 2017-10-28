import numpy as np

class Cell(object):
    def __init__(self, bias = 0.0, output = 0.0):
        self.correct = None
        self.output = output
        self.weights = [] # incoming weights from previous layer
    def set_output(self, output):
        self.output = output
    def init_weights(self, num_weights, weight_range=(-0.5, 0.5)):
        """
            You can think of bias term as the m'th incoming node
            weights spanning 1-m-1, the mth term is x_m = -1
            and the weight corresponding to it would be the bias term
        """
        low, high = weight_range
        self.bias = np.random.uniform(low, high)
        self.weights = np.random.uniform(low, high, num_weights)
    def reset_weights(self, weight_range):
        low, high = weight_range

        self.bias = np.random.uniform(low, high)
        self.weights = np.random.uniform(low, high, len(self.weights))
