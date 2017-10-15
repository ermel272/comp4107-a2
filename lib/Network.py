from scipy.misc import derivative
from Layer import Layer
class Network(object):
    def __init__(self, layers = [], learning_rate = 0.5):
        self.learning_rate = learning_rate
        self.layers = layers

    def feed_input (self, image_vector):
        assert len(self.layers[0].cells) == len(image_vector), """
            Number of cells in input layer should match length of input vector
        """
        normalized_image_vector = [1.0 if i else 0.0 for i in image_vector]
        for i in range(len(self.layers[0].cells)):
            self.layers[0].cells[i].set_output(normalized_image_vector[i])

    def feed_forward_network(self):
        for i in range(1, len(self.layers)): # for all layers after input layer
            p = self.layers[i - 1]
            l = self.layers[i]
            for j in range(len(l.cells)): # for each cell in current layer
                l.cells[j].output = l.cells[j].bias

                for w in range(len(l.cells[j].weights)):
                    l.cells[j].output += float(p.cells[w].output * l.cells[j].weights[w])
                # Process nodes output through activation function (def: Sigmoid)
                l.cells[j].output = l.activation_function(l.cells[j].output)


    def back_propogate(self, target_label = 0):
        """
            After performing feedforward, we have to
            find an error at each layer, and push it back
            and correct each layer.

            Params:
                target_label:Number - Expected output
        """
        normalized_target = 0 if target_label is 0 else 1
        for i in range(1, len(self.layers)):
            prev_layer = self.layers[-(i+1)]
            layer = self.layers[-i]
            for cell in layer.cells:
                errorsig = float((normalized_target - cell.output)) * derivative(layer.activation_function, cell.output, dx=1e-4)
                for w in range(len(cell.weights)):
                    cell.weights[w] += self.learning_rate * prev_layer.cells[w].output * errorsig
                cell.bias += self.learning_rate * errorsig


    def train(self, image_vector, image_label):
        """
            1. Feed image data into the network
            2. Calculate node outputs of *hidden* and *output* layers (=FEED FORWARD)
            3. Back-propagate the error and adjust the weights (=FEED BACKWARD)
            4. Classify the image (*guess* what digit is presented in the image)
        """
        assert len(self.layers) > 0, "No input layer has been defined"
        self.feed_input(image_vector)
        self.feed_forward_network()
        self.back_propogate(image_label)

    def identify(self, image_vector):
        assert len(self.layers) > 0, "No input layer has been defined"
        self.feed_input(image_vector)
        self.feed_forward_network()

        output_layer = self.layers[-1]
        max_so_far = 0
        i = -1

        for cell_index in range(len(output_layer.cells)):
            if output_layer.cells[cell_index].output > max_so_far:
                max_so_far = output_layer.cells[cell_index].output
                i = cell_index
        return i

    def add_layer(self, num_cells = 0, af = None):
        l = Layer(num_cells = num_cells, af=af)
        if len(self.layers) is not 0:
            l.init_weights(self.layers[-1].num_cells)
        self.layers.append(l)
