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

    input_training = train_data.reshape(60000, 784)[:500]
    distance_list = []
    whitened = whiten(input_training)
    k_values = range(1, 401)
    for i in k_values:
        print "Performing K-Means clustering with K = {}".format(i)
        centroids, distortion = kmeans(whitened, i)
        clusters = [list() for centroid in centroids]

        # Sort vector's into their clusters
        for vector in input_training:
            c = __find_closest_centroid(vector, centroids)
            clusters[c].append(vector)

        total_dist = 0
        # Compute distance from each cluster's points to their centroid
        for j in range(0, len(clusters)):
            centroid_dist = 0

            for vector in clusters[j]:
                centroid_dist += norm(vector - centroids[j]) ** 2

            total_dist += centroid_dist

        print "Objective centroid distance of {} observed".format(total_dist)
        distance_list.append(total_dist)

    plot = {"Distances": distance_list}
    fig, ax = plt.subplots()
    errors = pd.DataFrame(plot)
    errors.plot(ax=ax)
    plt.show()


def __find_closest_centroid(vector, centroids):
    centroid = 0
    smallest_distance = norm(vector - centroids[0]) ** 2

    for i in range(1, len(centroids)):
        dist = norm(vector - centroids[i]) ** 2

        if dist < smallest_distance:
            smallest_distance = dist
            centroid = i

    return centroid


if __name__ == '__main__':
    main()
