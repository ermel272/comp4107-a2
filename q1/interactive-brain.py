import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BRAIN = '.cache/brain.pickle'

net = pickle.load(open(BRAIN))

print 'Inspecting %s' % BRAIN
print 'Mean accuracy:', net.mean_accuracy
print 'Accuracy per training fold:', net.plot
images, labels = pickle.load(open('.cache/train-images-idx3-ubyte.pickle')), pickle.load(open('.cache/train-labels-idx1-ubyte.pickle'))
images = [image.flatten() for image in images]

def show(index):
    plt.imshow(np.array(images[index]).reshape(28, 28))
    plt.show()

def test(index, plt=False):
    if plt:
        show(index)
    return net.identify(images[index]), labels[index]

def accuracy():
    plot = {"Accuracy": net.accuracy_list}
    print 'mean_accuracy', net.mean_accuracy
    fig, ax = plt.subplots()
    errors = pd.DataFrame(plot)
    errors.plot(ax=ax)
    plt.show()

def test_some(start, end):
    accuracy = 0.
    for i in range(start, end):
        predicted, output = test(i)
        accuracy += int(predicted == output)
    accuracy /= (end - start)
    return accuracy
