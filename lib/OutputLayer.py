from Layer import Layer
class OutputLayer(Layer):
    def back_propagate(self, prev_layer, target, learning_rate = 0.5):
        for cell_index in range(len(self.cells)):
            normalized_target = 1 if cell_index == target else 0
            cell = self.cells[cell_index]
            errorsig = float(normalized_target - cell.output) * self.activation_function(cell.output)
            for w in range(len(cell.weights)):
                cell.weights[w] += float(learning_rate * prev_layer.cells[w].output * errorsig)
            cell.bias += learning_rate * errorsig
