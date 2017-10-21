from q1.lib.Network import Network
from math import e
from numpy.linalg import norm


def gaussian(x, beta, mu):
    """
    RBF Gaussian activation function.

    See http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
    """
    return e**(-beta*(norm(x-mu)**2))


class RBFNetwork(Network):
    pass
