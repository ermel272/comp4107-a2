from __future__ import division

from Layer import Layer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import pandas as pd

def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def sigmoid(x, der=False):
    """Logistic sigmoid function.
    Use der=True for the derivative."""
    if not der:
        return 1 / (1 + np.exp(-x))

class Network(object):
    def __init__(self, layers = [], learning_rate = 0.5, n_splits = 10):
        self.learning_rate = learning_rate
        self.n_splits = n_splits
        self.layers = layers

    def feed_input (self, image_vector = []):
        assert len(self.layers[0].cells) == len(image_vector), """
            Number of cells in input layer should match length of input vector
        """
        # experiment with this as only 1's and 0's
        normalized_image_vector = [float(i != 0) for i in image_vector]
        for i in range(len(self.layers[0].cells)):
            self.layers[0].cells[i].output = normalized_image_vector[i]
    def feed_forward_network(self):
        for i in range(1, len(self.layers)): # for all layers after input layer
            prev_layer = self.layers[i - 1]
            current_layer = self.layers[i]

            current_layer.reset_outputs()
            current_layer.reset_correct()

            for j in range(len(current_layer.cells)): # for each cell in current layer
                # z = w*a + b

                for w in range(len(current_layer.cells[j].weights)):
                    # Number of cells in previous layer is 1-1 with number of weights per cell in current layer
                    current_layer.cells[j].output += float(prev_layer.cells[w].output * current_layer.cells[j].weights[w])
                current_layer.cells[j].output += -1 * current_layer.cells[j].bias
                # Process nodes output through activation function
                # \sigma(z)
                current_layer.cells[j].output = current_layer.activation_function(current_layer.cells[j].output)
    def back_propagate(self, target_vector):
        """
            After performing feedforward, we have to
            find an error at each layer, and push it back
            and correct each layer.

            Params:
                target_label:Number - Expected output
        """
        output_layer = self.layers[-1]
        for i in range(len(output_layer.cells)):
            cell = output_layer.cells[i]

            cell_error = target_vector[i] - cell.output
            output_layer.cells[i].correct = cell_error * cell.output * (1 - cell.output)

        self.correct_network()
        self.update_weights()
    def update_weights(self):
        for layer_index in range(1, len(self.layers)):
            layer = self.layers[-layer_index]
            layer_before = self.layers[-1 * (layer_index + 1)]
            for cell_index in range(len(layer.cells)):
                cell = layer.cells[cell_index]
                # basically the dot product
                for w in range(len(cell.weights)):
                    layer.cells[cell_index].weights[w] += self.learning_rate * layer_before.cells[w].output * cell.correct
                # we should probably update bias too, because its also considered a weight
                layer.cells[cell_index].bias += self.learning_rate * 1 * cell.correct # 1 representing cell output

    def correct_network(self):
        """
            let k denote the layer #
            let j denote the cell (or neuron) index within layer k

            correct_{j}^{k} = out_{j}^{k} * (1 - out_{j}^{k}) * sum_{i}(all weights into node_{j}^{k}) * correct_{j}^{k+1}
        """
        # We already computed output layer at this point, and we won't modify correct term
        # for the input layer
        for layer_index in range(1, len(self.layers)):
            layer = self.layers[-layer_index]
            layer_before = self.layers[-1 * (layer_index + 1)]
            for cell_index in range(len(layer_before.cells)):
                cell = layer_before.cells[cell_index]
                term = sum([cell.weights[cell_index] * cell.correct for cell in layer.cells])
                layer_before.cells[cell_index].correct = cell.output * (1 - cell.output) * term

    def target_label_as_vector(self, target_label = 0):
        target_vector = np.zeros(len(self.layers[-1].cells))
        target_vector[target_label] = 1
        return target_vector


    def train(self, tset = [], tlabels = []):
        """
            1. Feed image data into the network
            2. Calculate node outputs of *hidden* and *output* layers (=FEED FORWARD)
            3. Back-propagate the error and adjust the weights (=FEED BACKWARD)
            4. Classify the image (*guess* what digit is presented in the image)
        """
        assert len(self.layers) > 0, "No input layer has been defined"
        self.accuracy_list = []
        gym = zip(tset, tlabels)
        random.shuffle(gym)
        # target_vector = self.target_label_as_vector(target_label)
        kfold = KFold(n_splits=self.n_splits)
        count = 0
        for training_indices, testing_indices in kfold.split(gym):
            training_set = [gym[i] for i in training_indices]
            testing_set = [gym[i] for i in testing_indices]
            for image, label in training_set:
                self.feed_input(image)
                self.feed_forward_network()
                self.back_propagate(self.target_label_as_vector(label))
                sys.stdout.write(".")
                sys.stdout.flush()

            num_correct = 0
            for test_image, test_label in testing_set:
                self.feed_input(test_image)
                self.feed_forward_network()
                prediction = self.identify(test_image)
                num_correct += int(prediction == test_label)
                sys.stdout.write(",")
                sys.stdout.flush()

            accuracy = num_correct / len(testing_set)
            self.accuracy_list.append(accuracy)
            self.mean_accuracy = np.mean(self.accuracy_list)

            count += 1

            sys.stdout.write("\n%d / %d folds completed.\n" % (count, kfold.get_n_splits()))
            sys.stdout.flush()

            sys.stdout.write("\n%.2f accuracy so far.\n" % self.mean_accuracy)
            sys.stdout.flush()

            if (self.mean_accuracy >= 0.8):
                break

        self.plot = {"Accuracy": self.accuracy_list}
        print 'mean_accuracy', self.mean_accuracy
        fig, ax = plt.subplots()
        errors = pd.DataFrame(self.plot)
        errors.plot(ax=ax)
        plt.show()

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
