import scipy.io as sio
import numpy as np
import math

class FixationHistoryMap:
    def __init__(self, h, w, hPadded, wPadded, settings):
        self.height = h
        self.width = w
        self.hPadded = hPadded
        self.wPadded = wPadded
        self.iorSizePx = int(settings.iorSizeDeg*settings.pix2deg)
        self.settings = settings

        self.fixHistMap = np.zeros((self.height, self.width), dtype=np.float32)
        self.fixHistMapPadded = np.zeros((self.hPadded, self.wPadded), dtype=np.float32)

        self.lastFixation = None
        self.fixationList = np.empty((0, 2), dtype=np.int32)


    def saveFixationCoords(self, fixCoords):
        self.lastFixation = fixCoords
        fixCoordsPadded = fixCoords + np.array([self.height/2, self.width/2], dtype=np.int32)

        self.fixationList = np.append(self.fixationList, [fixCoords], axis=0)
        #add inhibition area around the new fixation with radius iorSizePx
        for i in range(fixCoordsPadded[0]-self.iorSizePx, fixCoordsPadded[0]+self.iorSizePx):
            for j in range(fixCoordsPadded[1]-self.iorSizePx, fixCoordsPadded[1]+self.iorSizePx):
                d = math.sqrt((fixCoordsPadded[0]-i)*(fixCoordsPadded[0]-i) + (fixCoordsPadded[1]-j)*(fixCoordsPadded[1]-j))
                if d <= self.iorSizePx:
                    self.fixHistMapPadded[i,j] = min(1, self.fixHistMapPadded[i,j] + 1 - d/self.iorSizePx)

    def decayFixations(self):
        self.fixHistMapPadded -= 1/self.settings.iorDecayRate
        self.fixHistMapPadded = np.fmax(self.fixHistMapPadded, np.zeros((self.hPadded, self.wPadded)))

    def getFixationHistoryMap(self):
        #when there is no history of fixations yet return map of 0s
        if self.lastFixation is None:
            return self.fixHistMap
        else:
            return self.fixHistMapPadded[self.lastFixation[0]:self.lastFixation[0]+self.height, self.lastFixation[1]:self.lastFixation[1]+self.width]

    def dumpFixationsToMat(self, savePath):
        fixationList = np.fliplr(self.fixationList).astype(np.float64) # flip array since save format is [horz_coord, vert_coord]
        sio.savemat(savePath, {'fixations': fixationList})
