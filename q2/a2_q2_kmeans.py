#!/usr/bin/env python
"""
The purpose of this file is to perform the elbow method of finding the correct K value
for use within the RBF Network.

Note: This code will take a very long time to execute
"""
import json
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from scipy.cluster.vq import kmeans, whiten
from a2_q2 import maybe_download, extract, load_pickle


def main():
    with(open('config.json', 'r')) as f:
        config = json.load(f)

    train_filename_gz = maybe_download(config['train']['images'], 9912422)
    train_pickle = extract(train_filename_gz)
    train_data = load_pickle(train_pickle)

    # Only take the first 2000 data as a sample
    input_training = train_data.reshape(60000, 784)[:2000]
    distance_list = []
    whitened = whiten(input_training)
    max_k = 30
    for i in range(1, max_k + 1):
        print "Performing K-Means clustering with K = {}".format(i)
        centroids, distortion = kmeans(whitened, i)
        total_dist = 0

        # Compute the distance from all data points to their respective centroids
        for vector in input_training:
            dist = __find_closest_centroid_distance(vector, centroids)
            total_dist += dist

        mean_dist = total_dist / len(input_training)
        print "Mean distance of {} observed".format(mean_dist)
        distance_list.append(mean_dist)

    plot = {"Distances": distance_list}
    fig, ax = plt.subplots()
    errors = pd.DataFrame(plot)
    errors.plot(ax=ax)
    plt.show()


def __find_closest_centroid_distance(vector, centroids):
    smallest_distance = norm(vector - centroids[0])

    for i in range(1, len(centroids)):
        dist = norm(vector - centroids[i])
        smallest_distance = dist if dist < smallest_distance else smallest_distance

    return smallest_distance


if __name__ == '__main__':
    main()
