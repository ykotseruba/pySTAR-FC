from os import listdir
import os
import numpy as np
import cv2

import time

from Environment import Environment
from PeripheralAttentionalMap import PeripheralAttentionalMap
from CentralAttentionalMap import CentralAttentionalMap
from ConspicuityMap import ConspicuityMap
from PriorityMap import PriorityMap
from FixationHistoryMap import FixationHistoryMap
from Eye import Eye

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


class Controller:
    def __init__(self, settings):
        self.env = None
        self.eye = None

        self.settings = settings
        self.imageList = []

        self.periphMap=None
        self.centralMap=None
        self.conspMap=None
        self.priorityMap=None
        self.fixHistMap=None

        self.model_setup_done = False

        #save results
        self.saveResults = False
        if self.settings.saveFix:
            if os.path.exists(self.settings.saveDir):
                if self.settings.overwrite:
                    self.saveResults = True
            else:
                os.makedirs(self.settings.saveDir)
                self.saveResults = True

    #get input images
    def getInputImages(self):
        if self.settings.input:
            self.imageList.append(self.settings.input)
        else:
            #list all images in the directory
            self.imageList = [f for f in listdir(self.settings.batch) if any(f.endswith(ext) for ext in ['jpg', 'bmp', 'png', 'gif']) ]


    def setup(self, imgPath):
        imgName, ext = os.path.splitext(os.path.basename(imgPath))
        self.imgName = imgName
        self.env = Environment(self.settings)
        if self.settings.batch:
            self.env.loadStaticStimulus(self.settings.batch + '/' + imgPath)
        else:
            self.env.loadStaticStimulus(imgPath)
        self.eye = Eye(self.settings, self.env)

        if not self.model_setup_done:
            self.periphMap = PeripheralAttentionalMap(self.env.height, self.env.width, self.settings)
            self.centralMap = CentralAttentionalMap(self.env.height, self.env.width, self.settings)
            self.model_setup_done = True
        else:
            self.periphMap.update(self.env.height, self.env.width, self.settings)
            self.centralMap.update(self.env.height, self.env.width, self.settings)


        self.conspMap = ConspicuityMap(self.env.height, self.env.width, self.settings)
        self.priorityMap = PriorityMap(self.env.height, self.env.width, self.settings)
        self.fixHistMap = FixationHistoryMap(self.env.height, self.env.width, self.env.hPadded, self.env.wPadded, self.settings)

    #computes fixations for each image and each subject
    def run(self):
        self.getInputImages()
        for imgPath in self.imageList:

            for i in range(self.settings.numSubjects):
                self.setup(imgPath)
                self.computeFixations()

                if self.saveResults:
                    currentSaveDir = '{}/{}/'.format(self.settings.saveDir, self.imgName)
                    if not os.path.exists(currentSaveDir):
                        os.makedirs(currentSaveDir)
                    self.fixHistMap.dumpFixationsToMat('{}/fixations_{}.mat'.format(currentSaveDir, self.imgName, i))
                    cv2.imwrite('{}/fixations_{}.png'.format(currentSaveDir, self.imgName), self.env.sceneWithFixations.astype(np.uint8))

    def computeFixations(self):

        for i in range(self.settings.maxNumFixations):
            t0 = time.time()
            self.eye.viewScene()
            t_fov = time.time() - t0
            print('[FOVEATE]', self.eye.gazeCoords)
            print('[FOVEATE] Time elapsed {:0.03f}'.format(t_fov))

            prevGazeCoords = self.eye.gazeCoords.copy()

            t0 = time.time()
            self.periphMap.computeBUSaliency(self.eye.viewFov)
            self.periphMap.computePeriphMap(self.settings.blendingStrategy==1)
            t_periph = time.time() - t0
            print('[PeriphMap] Time elapsed {:0.03f}'.format(t_periph))

            t0 = time.time()
            self.centralMap.centralDetection(self.eye.viewFov)
            self.centralMap.maskCentralDetection()
            t_central = time.time() - t0
            print('[CentralMap] Time elapsed {:0.03f}'.format(t_central))

            #self.conspMap.computeConspicuityMap(self.periphMap.periphMap, self.centralMap.centralMap) #this is not used anywhere, for now commenting out

            t0 = time.time()
            self.fixHistMap.saveFixationCoords(prevGazeCoords)
            t_save = time.time() - t0
            print('[SaveFix] Time elapsed {:0.03f}'.format(t_save))

            t0 = time.time()
            self.priorityMap.computeNextFixationDirection(self.periphMap.periphMap, self.centralMap.centralMap, self.fixHistMap.getFixationHistoryMap())
            t_priority = time.time() - t0
            print('[PriorityMap] Time elapsed {:0.03f}'.format(t_priority))

            print('PrevGazeCoords=[{}, {}]'.format(prevGazeCoords[0], prevGazeCoords[1]))
            self.eye.setGazeCoords(self.priorityMap.nextFixationDirection)

            self.env.drawFixation(self.eye.gazeCoords.astype(np.int32), prevGazeCoords)

            t0 = time.time()
            self.fixHistMap.decayFixations()
            t_ior = time.time() - t0
            print('[IOR] Time elapsed {:0.03f}'.format(t_ior))

            if self.settings.visualize:
                t0 = time.time()
                if i == 0:
                    #plt.close('all')
                    plt.ion()
                    fig = plt.figure(1, figsize=(13,7), facecolor='white')
                    gs = gridspec.GridSpec(2, 3)
                    plt.show()
                    
                fig.clf()
                axes = []
                axes.append(self.add_subplot(fig, cv2.cvtColor(self.eye.viewFov, cv2.COLOR_BGR2RGB), 'Foveated View', gs[0,0]))
                axes.append(self.add_subplot(fig, self.periphMap.periphMap, 'Peripheral Map: ' + self.settings.PeriphSalAlgorithm, gs[0,1]))
                axes.append(self.add_subplot(fig, self.centralMap.centralMap, 'Central Map: ' + self.settings.CentralSalAlgorithm, gs[1,0]))
                axes.append(self.add_subplot(fig, self.priorityMap.priorityMap, 'Priority Map', gs[1,1]))
                axes.append(self.add_subplot(fig, cv2.cvtColor(self.env.sceneWithFixations.astype(np.uint8), cv2.COLOR_BGR2RGB), 'Image: {} \n Fixation #{}/{}'.format(self.imgName, i+1, self.settings.maxNumFixations), gs[:,-1]))
                gs.tight_layout(fig)
                fig.canvas.draw()
                t_vis = time.time() - t0
                print('[vis] Time elapsed {:0.03f}'.format(t_vis))
                plt.pause(0.01)

    def add_subplot(self, fig, img, title, plot_idx):
        ax = fig.add_subplot(plot_idx)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('[{:10.3f}, {:10.3f}]'.format(np.min(img), np.max(img)))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(img)
        return ax
