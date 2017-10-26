import math
from Cell import Cell
class Layer(object):
    def __init__(self, num_cells, af):
        self.num_cells = num_cells
        self.cells = [Cell() for i in range(num_cells)]
        self.activation_function = af

    def init_weights(self, num_cells, weight_range=(-0.5, 0.5)):
        self.num_weights = num_cells

        for i in range(len(self.cells)):
            self.cells[i].init_weights(num_cells, weight_range)

    def reset_weights(self):
        for i in range(len(self.cells)):
            self.cells[i].reset_weights
    def reset_outputs(self):
        for i in range(len(self.cells)):
            self.cells[i].output = 0
    def reset_correct(self):
        for i in range(len(self.cells)):
            self.cells[i].correct = None
