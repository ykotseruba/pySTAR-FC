import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom
from scipy.misc import logsumexp, imsave
import cv2
import time

class ICF:
    def __init__(self):
        #self.centerbias_template = np.load('contrib/ICF/centerbias.npy')
        check_point = 'contrib/ICF/ICF.ckpt'
        tf.reset_default_graph()
        new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))

        self.input_tensor = tf.get_collection('input_tensor')[0]
        self.centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
        self.log_density = tf.get_collection('log_density')[0]

        self.sess = tf.Session()
        new_saver.restore(self.sess, check_point)


    def loadImage(self, img):
        self.img = img.copy()
        #centerbias = zoom(self.centerbias_template, (self.img.shape[0]/1024, self.img.shape[1]/1024), order=0, mode='nearest')
        #centerbias -= logsumexp(centerbias)
        #self.centerbias_data = centerbias[np.newaxis, :, :, np.newaxis]
        self.centerbias_data = np.zeros([1, self.img.shape[0], self.img.shape[1], 1], dtype=np.float32)

    def computeSaliency(self):
        image_data = self.img[np.newaxis, :, :, :]
        log_density_prediction = self.sess.run(self.log_density, {
            self.input_tensor: image_data,
            self.centerbias_tensor: self.centerbias_data,
        })
        log_density = log_density_prediction[0, :, :, 0]
        #convert to traditional saliency map from log-probability
        self.sm = np.exp(log_density)
        self.sm /= np.sum(self.sm)
        cv2.normalize(self.sm, self.sm, 0, 1, cv2.NORM_MINMAX)

        return self.sm
