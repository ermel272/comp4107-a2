#!/usr/bin/env python
from lib.Cell import Cell
from lib.Layer import Layer
from lib.Target import Target
import matplotlib.pyplot as plt
import numpy as np

from scipy import misc
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sklearn.model_selection import KFold
from IPython.display import display, Image

import math, json, random, struct, array, os, operator, sys, functools, gzip

DATA_TYPES = {
    0x08: 'B',  # unsigned byte
    0x09: 'b',  # signed byte
    0x0b: 'h',  # short (2 bytes)
    0x0c: 'i',  # int (4 bytes)
    0x0d: 'f',  # float (4 bytes)
    0x0e: 'd'
}  # double (8 bytes)

LEARNING_RATE = 0.5
CACHE_DIR = '.cache'
URL  = 'http://yann.lecun.com/exdb/mnist/'

last_percent_reported = None

def main ():
    with(open('config.json', 'r')) as f:
        config = json.load(f)

    train_filename_gz = maybe_download(config["train"]["images"], 9912422)
    test_filename_gz  = maybe_download(config["test"]["images"], 1648877)
    train_labels_gz = maybe_download(config["train"]["labels"], 28881)

    train_pickle = extract(train_filename_gz)
    train_labels_pickle = extract(train_labels_gz)
    test_pickle = extract(test_filename_gz)

    train_data = load_pickle(train_pickle)
    train_labels = load_pickle(train_labels_pickle)
    test_data = load_pickle(test_pickle)
    # There are now 60,000 items of length 784 (28x28)
    # This will serve as input to neural network
    # Each cell will have 784 inputs
    input_training = [i.flatten() for i in train_data]

    print len(input_training)
    print len(train_labels)

    # Initialize our cells as empty
    cells = [Cell(28 * 28, learning_rate=LEARNING_RATE) for i in range(10)]
    # Since we want to recognize 10 different handwritten digits, our network
    # needs 10 cells per layer (for digits 0-9)

    # our system will be simple, one hidden layer
    layer1 = Layer(cells)

    layers = [layer1]
    # training
    for image, output in zip(input_training, train_labels)[:500]:
        target = Target(output)
        for i in range(len(layer1.cells)):
            layer1.cells[i].set_inputs(image)
            actual_output = layer1.cells[i].compute_and_update_output()

            error_delta = target.get(i) - actual_output
            layer1.cells[i].update_weights(error_delta)

    test_image, test_output = zip(input_training, train_labels)[50]

    for i in range(len(layer1.cells)):
        layer1.cells[i].set_inputs(test_image)
        layer1.cells[i].compute_and_update_output()

    print "Predicted: %s" % layer1.predict()
    print "Actual: %s" % test_output

    test_image, test_output = zip(input_training, train_labels)[51]

    for i in range(len(layer1.cells)):
        layer1.cells[i].set_inputs(test_image)
        layer1.cells[i].compute_and_update_output()

    print "Predicted: %s" % layer1.predict()
    print "Actual: %s" % test_output


    # Target (output)
    # The target output will be a binary representation of our output


    # print train_labels
    #
    # kf = KFold(n_splits = 10)
    #
    # for train_index, test_index in kf.split(train_data):
    #     break


def download_progress_hook(count, blockSize, totalSize):
  global last_percent_reported

  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent

def maybe_download (filename, expected_bytes, force=False):
    dest_filename = os.path.join(CACHE_DIR, filename)
    if force or not os.path.exists(dest_filename):
        print 'Attempting to download: %s' % filename
        filename, _ = urlretrieve(URL + filename, dest_filename, reporthook=download_progress_hook)
        print 'Download Complete!'
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print 'Found and verified %s' % dest_filename
    else:
        raise Exception('Failed to verify ' + dest_filename)
    return dest_filename

def extract(filename, force=False):
    with gzip.open(filename, 'rb') as fd:
        pickle_file = filename.replace('.gz', '.pickle')
        return maybe_pickle(pickle_file, parse_idx(fd), force)


def maybe_pickle(filename, data, force=False):
    if (not os.path.exists(filename) or force):
        with open(filename, 'wb') as pf:
            print 'Pickling %s' % filename
            pickle.dump(data, pf, pickle.HIGHEST_PROTOCOL)
    return filename

def parse_idx(fd):
    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))
    return np.array(data).reshape(dimension_sizes)

def load_pickle (f):
    print 'Performing pickle.load(%s)' % f
    with open(f, 'rb') as tp:
        return pickle.load(tp)

def sigmoid(x):
  return 1.0 / (1 + math.exp(-x))


if __name__ == '__main__':
    main()