import numpy as np

class ConspicuityMap:
    def __init__(self, h, w, settings):
        self.height = h
        self.width = w
        self.settings = settings
        self.conspMap = None

    def computeConspicuityMap(self, periphMap, centralMap):

        periphMap *= self.settings.pgain
        centralMap *= self.settings.cgain

        self.conspMap = np.fmax(periphMap, centralMap)
