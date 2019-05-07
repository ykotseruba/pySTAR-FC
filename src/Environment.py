import math
from enum import Enum

import numpy as np
import cv2


class Environment:

    def __init__(self, settings):

        self.settings = settings

        if self.settings.inputSizeDeg:
            self.widthm = 2*self.settings.viewDist*math.tan((self.settings.inputSizeDeg*math.pi/180)/2)
            self.dotPitchMethod = 'FROM_VIS_ANGLE_SIZE'
        else:
            self.dotPitchMethod = 'FROM_DEG2PIX'

        self.scene = None
        self.sceneWithFixations = None
        self.width = -1
        self.height = -1
        self.wPadded = -1
        self.hPadded = -1
        self.prevFix = None
        self.dotPitch = -1


    def loadStaticStimulus(self, imgPath):
        self.scene = cv2.imread(imgPath)

        if self.scene is None:
            raise IOError('Cannot open image {0}!'.format(imgPath))

        self.sceneWithFixations = self.scene.astype(np.float32).copy()
        self.height, self.width, self.depth = self.scene.shape
        self.prevFix = np.array([self.height/2, self.width/2], dtype=np.int32)

        self.padStaticStimulus()
        self.updateDotPitch()

    def padStaticStimulus(self):
        if self.settings.paddingRGB[0] < 0:
            self.settings.paddingRGB = self.scene.mean(axis=(0,1))

        self.scenePadded = cv2.copyMakeBorder(self.scene, round(self.height/2), round(self.height/2), round(self.width/2), round(self.width/2), cv2.BORDER_CONSTANT, value=self.settings.paddingRGB.astype(np.float64))

        self.wPadded = self.scenePadded.shape[1]
        self.hPadded = self.scenePadded.shape[0]


    def updateDotPitch(self):
        if self.dotPitchMethod == 'FROM_VIS_ANGLE_SIZE':
            self.dotPitch = self.widthm/self.width;
            self.settings.pix2deg = self.width/self.settings.inputSizeDeg;
        elif self.dotPitchMethod == 'FROM_DEG2PIX':
            self.settings.inputSizeDeg = self.width/pix2deg;
            self.widthm = 2*self.settings.viewDist*math.tan(self.settings.inputSizeDeg/2)
            self.dotPitch = self.widthm/self.width
        else:
            raise ValueError('Unrecognized option for updateDotPitch!')

    def getEyeView(self, gazeCoords):
        return self.scenePadded[gazeCoords[0]:gazeCoords[0]+self.height, gazeCoords[1]:gazeCoords[1]+self.width, :]

    def drawFixation(self, newFix):
        cv2.line(self.sceneWithFixations, (self.prevFix[1], self.prevFix[0]), (newFix[1], newFix[0]),(62, 62, 250), 2, cv2.LINE_AA)
        cv2.circle(self.sceneWithFixations, (self.prevFix[1], self.prevFix[0]), 1, (62, 62, 250), cv2.LINE_AA)
        cv2.circle(self.sceneWithFixations, (newFix[1], newFix[0]), 1, (62, 62, 250), cv2.LINE_AA)
        self.prevFix = newFix.copy();
