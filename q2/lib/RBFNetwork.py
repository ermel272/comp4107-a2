import random

import sys
import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean
from numpy.linalg import norm
from scipy.cluster.vq import kmeans, whiten
from sklearn.model_selection import KFold

from q2.lib.Neuron import Neuron


class RBFNetwork(object):
    def __init__(self, data, k=20, output=10, learning_rate=0.125):
        self.learning_rate = learning_rate
        self.hidden_layer = []
        self.output_layer = [i for i in range(0, output)]
        self.init_hidden_layer(k, data, output)

    def init_hidden_layer(self, k, data, output):
        centroids, distortion = kmeans(whiten(data), k)
        clusters = compute_clusters(centroids, data)

        for i in range(0, len(centroids)):
            sigma = compute_sigma(centroids[i], clusters[i])
            beta = compute_beta(sigma)
            self.hidden_layer.append(Neuron(output, beta, centroids[i]))

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
                # TODO: Implement calculation of estimate given neuron outputs
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
