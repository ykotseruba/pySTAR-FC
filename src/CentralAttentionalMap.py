import numpy as np
import time
import math
import cv2

class CentralAttentionalMap:
    def __init__(self, h, w, settings):
        self.height = h
        self.width = w
        self.settings = settings
        self.centralMask = None
        self.centralMap = None
        if 'DeepGazeII' in settings.CentralSalAlgorithm:
            from DeepGazeII import DeepGazeII
            self.buSal = DeepGazeII()
        if 'SALICONtf' in settings.CentralSalAlgorithm:
            from SALICONtf import SALICONtf
            self.buSal = SALICONtf()
        self.initCentralMask()

    def initCentralMask(self):
        self.centralMask = np.zeros((self.height,self.width), np.uint8)
        centX = round(self.height/2)
        centY = round(self.width/2)

        self.settings.cSizePix = self.settings.cSizeDeg*self.settings.pix2deg

        for i in range(self.height):
            for j in range(self.width):
                rad = math.sqrt((i-centX)*(i-centX) + (j-centY)*(j-centY))
                if (rad <= self.settings.cSizePix):
                    self.centralMask[i, j] = 1
                else:
                    self.centralMask[i, j] = 0


    def centralDetection(self, view):
        #SALICON works on images with range [0, 255]
        self.centralMap = self.buSal.compute_saliency(view*255)
        cv2.normalize(self.centralMap, self.centralMap, 0, 1, cv2.NORM_MINMAX)

    def maskCentralDetection(self):
        self.centralMap[self.centralMask == 0] = 0
        # cv2.imshow('image',self.centralMap)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
