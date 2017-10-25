import random

import sys
import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean, zeros, delete
from numpy.linalg import norm
from scipy.cluster.vq import kmeans, whiten
from sklearn.model_selection import KFold

from q2.lib.Neuron import Neuron


class RBFNetwork(object):
    def __init__(self, data, k=20, output=10, learning_rate=0.125):
        self.learning_rate = learning_rate
        self.hidden_layer = []
        self.output_layer = [0] * output
        self.init_hidden_layer(k, data, output, learning_rate)

    def init_hidden_layer(self, k, data, output, learning_rate):
        print "Performing K means on {} inputs with k={}".format(len(data), k)
        centroids, distortion = kmeans(whiten(data), k)
        clusters = compute_clusters(centroids, data)
        empty_clusters = list()

        # Filter out clusters with no data
        for i in range(0, len(clusters)):
            if len(clusters[i]) == 0:
                empty_clusters.append(i)

        for i in range(0, len(centroids)):
            if i in empty_clusters:
                continue

            sigma = compute_sigma(centroids[i], clusters[i])
            beta = compute_beta(sigma)
            self.hidden_layer.append(Neuron(output, beta, centroids[i], leaning_rate=learning_rate))

    def train(self, tset, tlabels):
        accuracy_list = []
        mean_accuracy = 0
        gym = zip(tset, tlabels)[:2000]
        random.shuffle(gym)

        kfold = KFold(n_splits=10)
        count = 0
        for training_indices, testing_indices in kfold.split(gym):
            training_set = [gym[i] for i in training_indices]
            testing_set = [gym[i] for i in testing_indices]

            for image, label in training_set:
                self.feed_input(image)
                self.forward_propagate()
                self.back_propagate(self.target_label_as_vector(label))
                sys.stdout.write(".")
                sys.stdout.flush()

            num_correct = 0
            for test_image, test_label in testing_set:
                self.feed_input(test_image)
                self.forward_propagate()
                prediction = self.identify(test_image)
                num_correct += int(prediction == test_label)
                sys.stdout.write(",")
                sys.stdout.flush()

            accuracy = num_correct / len(testing_set)
            accuracy_list.append(accuracy)
            mean_accuracy = mean(accuracy_list)

            count += 1

            print "{} / {} folds completed.".format(count, kfold.get_n_splits())
            print "{0:.2f} accuracy so far".format(mean_accuracy)

        plot = {"Accuracy": accuracy_list}
        print 'mean_accuracy', mean_accuracy
        fig, ax = plt.subplots()
        errors = pd.DataFrame(plot)
        errors.plot(ax=ax)
        plt.show()

    def feed_input(self, vector):
        # Pass the input vector to each neuron in the network
        for neuron in self.hidden_layer:
            neuron.update_output(vector)

    def forward_propagate(self):
        """
        Propagate the input vector forward through the neural network.
        """
        for i in range(0, len(self.output_layer)):
            output = 0

            # Loop through each Neuron in the hidden layer
            for neuron in self.hidden_layer:
                output += neuron.weights[i] * neuron.output

            # Update summation for output classifier
            self.output_layer[i] = output

    def back_propagate(self, label_vector):
        """
        Update hidden layer neuron weights based on output error.
        """
        for i in range(0, len(self.output_layer)):
            last_neuron_error = label_vector[i] - self.output_layer[i]

            # Update each neuron's correction value for the weight pointing at the specified output cell
            for neuron in self.hidden_layer:
                neuron.update_correction(last_neuron_error, i)

        # Apply corrections to each neuron
        for neuron in self.hidden_layer:
            neuron.apply_corrections()

    def target_label_as_vector(self, target_label=0):
        target_vector = zeros(len(self.output_layer))
        target_vector[target_label] = 1
        return target_vector

    def identify(self, image_vector):
        self.feed_input(image_vector)
        self.forward_propagate()
        max_so_far = 0
        i = -1

        for cell_index in range(len(self.output_layer)):
            if self.output_layer[cell_index] > max_so_far:
                max_so_far = self.output_layer[cell_index]
                i = cell_index
        return i


def find_closest_centroid(vector, centroids):
    centroid = 0
    smallest_distance = norm(vector - centroids[0]) ** 2

    for i in range(1, len(centroids)):
        dist = norm(vector - centroids[i]) ** 2

        if dist < smallest_distance:
            smallest_distance = dist
            centroid = i

    return centroid


def compute_clusters(centroids, data):
    clusters = [list() for i in centroids]

    # Sort vector's into their clusters
    for vector in data:
        c = find_closest_centroid(vector, centroids)
        clusters[c].append(vector)

    return clusters


def compute_sigma(centroid, data):
    """
    Computes sigma as per http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
    """
    dist = 0
    for vector in data:
        dist += norm(vector - centroid)

    return dist / len(data)


def compute_beta(sigma):
    return 1 / (2 * (sigma ** 2))
