import pickle
import matplotlib.pyplot as plt
import numpy as np

net = pickle.load(open('.cache/brain.pickle'))
images, labels = pickle.load(open('.cache/train-images-idx3-ubyte.pickle')), pickle.load(open('.cache/train-labels-idx1-ubyte.pickle'))
images = [image.flatten() for image in images]

def show(image):
    plt.imshow(np.array(image).reshape(28, 28))
    plt.show()

def test(index, plt=False):
    if plt:
        show(images[index])
    print net.identify(images[index]), labels[index]
