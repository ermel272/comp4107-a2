import numpy

class Target(object):
    def __init__(self, value):
        v = numpy.zeros(10)
        v[value] = 1.0
        self.value = v
    def __repr__(self):
        return repr(self.value)
    def get(self, index):
        return self.value[index]
