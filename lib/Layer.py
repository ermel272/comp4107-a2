class Layer(object):
    def __init__(self, cells = []):
        self.cells = cells
    def predict(self):
        maxOut = 0
        maxIndex = 0

        for i in range(len(self.cells)):
            if self.cells[i].output > maxOut:
                maxOut = self.cells[i].output
                maxIndex = i
        return maxIndex
