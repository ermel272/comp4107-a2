import numpy as np
class Cell(object):
    def __init__(self, bias = 0.0, output = 0.0):
        self.correct = None
        self.output = output
        self.weights = [] # incoming weights from previous layer
    def set_output(self, output):
        self.output = output
    def init_weights(self, num_weights):
        """
            You can think of bias term as the m'th incoming node
            weights spanning 1-m-1, the mth term is x_m = -1
            and the weight corresponding to it would be the bias term
        """
        self.bias = np.random.rand()
        self.weights = np.random.rand(num_weights)
