from Layer import Layer
class HiddenLayer(Layer):
    def back_propagate(self, prev_layer, next_layer, target, learning_rate = 0.5):
        for cell_index in range(len(self.cells)):
            cell = self.cells[cell_index]
            pcell_error = 0
            for pcell_index in range(len(prev_layer.cells)):
                normalized_target = 1 if pcell_index == target else 0

                pcell = prev_layer.cells[pcell_index]

                errorsig = float(normalized_target - pcell.output) * prev_layer.activation_function(pcell.output)
                pcell_error += errorsig * pcell.weights[cell_index];
            error_signal = pcell_error * self.activation_function(cell.output)
            for w in range(len(cell.weights)):
                cell.weights[w] += learning_rate * next_layer.cells[w].output * error_signal
            cell.bias += learning_rate * 1 * error_signal
