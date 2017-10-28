from __future__ import division

from Layer import Layer
from sklearn.model_selection import KFold
import numpy as np
import random
import sys
import dill

sigmoid = lambda x: 1. / (1 + np.exp(-x))

def save(net):
    layer_arch = "-".join([str(l.num_cells) for l in net.layers])
    filename = ".cache/brain--learning_rate=%s&tol=%s&max_iter=%s&n_splits=%s&layers=%s.pickle" % (net.learning_rate, net.tolerance, net.max_iter, net.n_splits, layer_arch)
    dill.dump(net, open(filename, 'wb'))
    print 'Saved file as %s' % filename

class Network(object):

    def __init__(self, layers = [], learning_rate = 0.5, n_splits = 10, weight_range=(-0.5, 0.5), max_iter=100, tolerance=0.01, max_no_improvements = 3):
        self.learning_rate = learning_rate
        self.weight_range = weight_range
        self.n_splits = n_splits
        self.layers = layers
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.max_no_improvements = max_no_improvements

    def sanitize_image(self, image_vector = []):
        return [float(pixel != 0) for pixel in image_vector]

    def feed_input (self, image_vector = []):
        assert len(self.layers[0].cells) == len(image_vector), """
            Number of cells in input layer should match length of input vector
        """
        # experiment with this as only 1's and 0's
        normalized_image_vector = self.sanitize_image(image_vector)
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
                o = np.dot(current_layer.cells[j].weights, map(lambda cell: cell.output, prev_layer.cells))
                o += -1 * current_layer.cells[j].bias

                current_layer.cells[j].output = current_layer.activation_function(o)
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
                term = sum([c.weights[cell_index] * c.correct for c in layer.cells])
                cell.correct = cell.output * (1 - cell.output) * term

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

        kfold = KFold(n_splits=self.n_splits)
        count = 0
        for training_indices, testing_indices in kfold.split(gym):
            self.reset_weights()
            training_set = [gym[i] for i in training_indices]
            testing_set = [gym[i] for i in testing_indices]

            pre_accuracy = 0
            accuracy = 0
            no_improvement_count = 0

            for i in range(self.max_iter):
                for image, label in training_set:
                    self.feed_input(image)
                    self.feed_forward_network()
                    self.back_propagate(self.target_label_as_vector(label))
                    # sys.stdout.write(".")
                    # sys.stdout.flush()

                # calculate cost
                accumulated_errors = []
                accuracy = 0

                for test_image, test_label in testing_set:
                    prediction = self.identify(test_image)
                    accuracy += int(test_label == prediction)

                    sys.stdout.write("*")
                    sys.stdout.flush()

                accuracy /= len(testing_set)

                print '\n(%.3f - %.3f) <= %.3f' % (accuracy, pre_accuracy, self.tolerance)
                if (accuracy - pre_accuracy) <= self.tolerance:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0

                print '\nMean accuracy so far, ', accuracy
                print 'No Improvement count: %d / %d' % (no_improvement_count, self.max_no_improvements)
                print '%d out of %d' % (i, self.max_iter)
                sys.stdout.write("\n%d / %d folds.\n" % (count, kfold.get_n_splits()))
                sys.stdout.flush()

                # we've converged (sort of)
                if no_improvement_count >= self.max_no_improvements:
                    print 'Converged!'
                    break

                pre_accuracy = accuracy

            self.accuracy_list.append(accuracy)

            self.mean_accuracy = np.mean(self.accuracy_list)
            count += 1

        self.plot = {"Accuracy": self.accuracy_list}
        self.confidence_interval = (
            self.mean_accuracy - 3 * np.std(self.accuracy_list),
            self.mean_accuracy + 3 * np.std(self.accuracy_list)
        )
        print 'confidence_interval', self.confidence_interval

    def identify(self, image_vector):
        self.feed_input(image_vector)
        self.feed_forward_network()
        return np.argmax(map(lambda cell: cell.output, self.layers[-1].cells))

    def add_layer(self, num_cells = 0, af = None):
        l = Layer(num_cells = num_cells, af=af, weight_range=self.weight_range)
        if len(self.layers) is not 0:
            l.init_weights(self.layers[-1].num_cells)

        self.layers.append(l)

    def reset_weights(self):
        for layer in self.layers:
            layer.reset_weights()
