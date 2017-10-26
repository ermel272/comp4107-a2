#!/usr/bin/env python

import numpy as np
from lib.Network import Network, sigmoid, net_as_string
from lib.util import *
# This is a simple example with one layer, it's not sufficient
# Just to kind of get started
def main():
    with(open('config.json', 'r')) as f:
        config = json.load(f)

    train_filename_gz = maybe_download(config['train']['images'], 9912422)
    test_filename_gz = maybe_download(config['test']['images'], 1648877)
    train_labels_gz = maybe_download(config['train']['labels'], 28881)
    test_labels_gz = maybe_download(config['test']['labels'], 4542)

    train_pickle = extract(train_filename_gz)
    train_labels_pickle = extract(train_labels_gz)
    test_pickle = extract(test_filename_gz)
    test_labels_pickle = extract(test_labels_gz)

    train_data = load_pickle(train_pickle)
    train_labels = load_pickle(train_labels_pickle)
    test_data = load_pickle(test_pickle)
    test_labels = load_pickle(test_labels_pickle)

    # There are now 60,000 items of length 784 (28x28)
    # This will serve as input to neural network
    # Each cell will have 784 inputs
    input_training = train_data.reshape(60000, 784)

    # our system will be simple, one hidden layer
    net = Network(
        learning_rate=.125,
        n_splits=10,
        max_epoch=100,
        tolerance=0.1,
        max_no_improvements=3,
        weight_range=(-0.5, 0.5))

    net.add_layer(784)

    net.add_layer(20, sigmoid)

    net.add_layer(10, sigmoid)  # output layer

    net.train(input_training, train_labels)
    net.save()

if __name__ == '__main__':
    main()
