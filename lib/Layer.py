import math
from Cell import Cell
class Layer(object):
    def __init__(self, num_cells, af):
        self.num_cells = num_cells
        self.cells = [Cell() for i in range(num_cells)]
        self.activation_function = af

    def init_weights(self, num_cells):
        for i in range(len(self.cells)):
            self.cells[i].init_weights(num_cells)
