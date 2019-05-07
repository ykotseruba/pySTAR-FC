import scipy.io as sio
import scipy.signal as sig
import numpy as np
import cv2
import time
import math

class AIM:
    def __init__(self, basisMatPath='data/21infomax900.mat'):

        self.scale = 1
        self.origH = -1
        self.origW = -1
        self.newH = -1
        self.newW = 700
        self.img = None
        self.aimTemp = None
        self.sm = None

        self.loadBasis(basisMatPath)

    def loadBasis(self, basisMatPath):
        B = sio.loadmat(basisMatPath)['B'].astype(np.float32)
        B = np.asfortranarray(B)
        kernel_size = int(math.sqrt(B.shape[1]/3))
        print(B.shape, kernel_size)
        self.basis = np.reshape(B, (B.shape[0], kernel_size, kernel_size, 3), order='F')

        #AIM requires correlation operation, but since scipy only has convolution available
        #we need to flip the kernels vertically and horizontally
        for i in range(self.basis.shape[0]):
            for j in range(self.basis.shape[3]):
                self.basis[i, :, :, j] = np.fliplr(np.flipud(self.basis[i, :, :, j]))


    def loadImage(self, img):
        self.img = img.copy()
        self.img = self.img.astype(np.float32)

        self.origH = img.shape[0]
        self.origW = img.shape[1]

        self.newH = int(img.shape[0]*(self.newW/img.shape[1]))
        self.img = cv2.resize(self.img, (self.newW, self.newH), interpolation=cv2.INTER_AREA)
        self.aimTemp = np.zeros((self.img.shape[0]-self.basis.shape[1]+1, self.img.shape[1]-self.basis.shape[1]+1, self.basis.shape[0]), dtype=np.float32)


    def computeSaliency(self):
        border = round(self.basis.shape[1]/2)

        imgCopy = self.img.copy()

        #AIM kernels are ordered for RGB channels
        #so convert the image from BGR to RGB
        imgCopy = imgCopy[...,::-1]
        #t0 = time.time()

        #convolve the image with kernels for each channel
        for f in range(self.basis.shape[0]):
            #TODO: replace with pyFFTW
            self.aimTemp[:, :, f] = sig.fftconvolve(imgCopy[:, :, 0], self.basis[f, :, :, 0], mode='valid')

            for c in range(1, self.img.shape[2]):
                temp = sig.fftconvolve(imgCopy[:, :, c], self.basis[f, :, :, c], mode='valid')
                self.aimTemp[:, :, f] += temp

        maxAIM = np.amax(self.aimTemp)
        minAIM = np.amin(self.aimTemp)

        #print('[AIM] minAIM=' + str(minAIM), 'maxAIM=' + str(maxAIM))

        #rescale image using global max and min
        for f in range(self.basis.shape[0]):
            self.aimTemp[:, :, f] -= minAIM
            self.aimTemp[:, :, f] /= (maxAIM - minAIM)

        #compute histograms for each feature map
        #and use them to rescale values based on histogram to reflect likelihood
        div = 1/(self.aimTemp.shape[0]*self.aimTemp.shape[1])

        self.sm = np.zeros((self.aimTemp.shape[0], self.aimTemp.shape[1]), dtype=np.float32)

        numBins = 256
        for f in range(self.basis.shape[0]):

            hist, bin_edges = np.histogram(self.aimTemp[:, :, f], bins=numBins, range=[0,1], density=False)
            idx = self.aimTemp[:, :, f]*(numBins-1)
            idx = idx.astype(int)
            self.sm -= np.log(hist[idx]*div+0.000001)


        cv2.normalize(self.sm, self.sm, 0, 1, cv2.NORM_MINMAX)
        self.sm = cv2.GaussianBlur(self.sm,(31, 31), 8, cv2.BORDER_CONSTANT)
        self.sm = cv2.copyMakeBorder(self.sm, border, border, border, border, cv2.BORDER_CONSTANT, 0)
        self.sm = cv2.resize(self.sm, (self.origW, self.origH), 0, 0, cv2.INTER_AREA)


        #t1 = time.time()
        #print('[AIM] Time elapsed {:1.03f}'.format(t1-t0))

        #cv2.imshow('image',self.sm*255)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return self.sm
