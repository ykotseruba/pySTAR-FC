import numpy as np
import cv2
import time
import cProfile


class PriorityMap:
    def __init__(self, h, w, settings):
        self.settings = settings
        self.height = h
        self.width = w
        self.priorityMap = np.zeros((h, w), dtype=np.float32)
        self.nextFixationDirection = (-1, -1)
        self.dist = None
        self.initDist()

    def reset(self, h, w):
        self.priorityMap = np.zeros((h, w), dtype=np.float32)
        self.height = h
        self.width = w
        self.nextFixationDirection = (-1, -1)
        self.initDist()

    #precompute euclidean distances to the center for every pixel in the image
    def initDist(self):
        centX = int(self.height/2)
        centY = int(self.width/2)
        x, step = np.linspace(0, self.height-1, num=self.height, retstep=True, dtype=np.float32)
        y, step = np.linspace(0, self.width-1, num=self.width, retstep=True, dtype=np.float32)
        iy, ix = np.meshgrid(y, x, sparse=False, indexing='xy')
        self.dist = np.sqrt(np.power(iy-centY, 2) + np.power(ix-centX, 2))

    #compute a weighted combination of the central and peripheral maps
    def combinePeriphAndCentralWeighted(self, periphMap, centralMap, fixHistMap):
        rC = self.settings.pSizePix
        rMax = np.amax(self.dist)
        # #TODO: this is taking 6 seconds, can use numpy here
        # for i in range(0, self.height):
        #     for j in range(0, self.width):
        #         rP = self.dist[i,j]
        #         if rP < rC:
        #             self.priorityMap[i, j] = ((rC-rP)/rC)*centralMap[i, j] + (rP/rC)*periphMap[i, j]
        #         else:
        #             self.priorityMap[i, j] = (1 + ((rP-rC)/(rMax-rC))*(abs(1-self.settings.pgain)))*periphMap[i, j]

        rP_lessthan_rC = self.dist < rC
        rP_grthan_rC = self.dist >= rC

        temp1 = np.multiply((rC-self.dist)/rC, centralMap) + np.multiply(self.dist/rC,*periphMap);
        temp2 = np.multiply(1 + ((self.dist-rC)/(rMax-rC))*(abs(1-self.settings.pgain)), periphMap);
        self.priorityMap = np.multiply(temp1, rP_lessthan_rC)+np.multiply(temp2, rP_grthan_rC);

        #print(np.allclose(prMap, self.priorityMap))
        #print((prMap == self.priorityMap).all())
        self.priorityMap -= fixHistMap

    #multiply peripheral and central map by pgain and cgain factors respectively
    #subtract fixation history map from scaled peripheral and central maps
    #compute element-wise maximum of the two inhibited maps
    def combinePeriphAndCentralMax(self, periphMap, centralMap, fixHistMap):
        self.priorityMap = np.fmax(periphMap*self.settings.pgain-fixHistMap, centralMap*self.settings.cgain-fixHistMap)

    #determine next fixation direction:
    #1. first combine central and peripheral map into priority map
    #2. find maximum of the priority map as either
    #   a. location of the maximum for deterministic fixations
    #   b. using a predefined threshold (e.g. 95%), binarize the priority map
    #   and select a single maximum point (uniform distribution)
    def computeNextFixationDirection(self, periphMap, centralMap, fixHistMap):
        #t0 = time.time()
        if self.settings.blendingStrategy == 3:
            self.combinePeriphAndCentralWeighted(periphMap, centralMap, fixHistMap)
        else:
            self.combinePeriphAndCentralMax(periphMap, centralMap, fixHistMap)
        #t1 = time.time()
        #print('[PRIORITY MAP] Time elapsed {:0.03f}'.format(t1-t0))
        cv2.normalize(self.priorityMap, self.priorityMap, 0, 1, cv2.NORM_MINMAX)

        if self.settings.nextFixAsMax:
            maxLoc = np.unravel_index(self.priorityMap.argmax(), self.priorityMap.shape)
            self.nextFixationDirection = np.array(maxLoc, dtype=np.int32) - np.array([self.height/2, self.width/2], dtype=np.int32)
        else:
            priorityMapCopy = self.priorityMap.copy()
            maxVal = np.amax(self.priorityMap)
            priorityMapCopy[priorityMapCopy >= maxVal*self.settings.nextFixThresh] = 1
            priorityMapCopy[priorityMapCopy < maxVal*self.settings.nextFixThresh] = 0
            nonzeroVals = np.flatnonzero(priorityMapCopy)
            s = np.random.uniform(0, nonzeroVals.shape[0], 1)
            self.nextFixationDirection = np.array(np.unravel_index(nonzeroVals[s], self.priorityMap.shape), dtype=np.int32) - np.array([self.height/2, self.width/2], dtype=np.int32)
