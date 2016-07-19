'''A classifier using keras

The model is composed of the json keras model and the weights
'''

import keras
from keras.optimizers import SGD
import numpy as np
import time

from rh_logger import logger
from ..targets.classifier_target import AbstractPixelClassifier


class KerasClassifier(AbstractPixelClassifier):
    
    def __init__(self, model_path, weights_path, 
                 xypad_size, zpad_size, block_size):
        '''Initialize from a model and weights
        
        :param model_path: path to JSON model file suitable for 
        keras.models.model_from_json
        :param weights_path: path to weights .h5 file
        :param xypad_size: padding needed in x and y
        :param zpad_size: padding needed in z
        :param block_size: size of input block to classifier
        '''
        self.xypad_size = xypad_size
        self.zpad_size = zpad_size
        self.block_size = block_size
        self.model = keras.models.model_from_json(
            open(model_path, "r").read())
        self.model.load_weights(weights_path)
        self.__finish_model()

    def __finish_model(self):
        '''Compile the model'''
        sgd = SGD(lr=0.01, decay=0, momentum=0.0, nesterov=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
        
    def __getstate__(self):
        '''Get the pickleable state of the model'''
        
        return dict(xypad_size=self.xypad_size,
                    zpad_size=self.zpad_size,
                    block_size=self.block_size,
                    model=self.model.to_json(),
                    weights=self.model.get_weights())
    
    def __setstate__(self, state):
        '''Restore the state from the pickle'''
        self.xypad_size = state["xypad_size"]
        self.zpad_size = state["zpad_size"]
        self.block_size = state["block_size"]
        self.model = keras.models.model_from_json(state["model"])
        self.model.set_weights(state["weights"])
        self.__finish_model()
    
    def get_class_names(self):
        return ["membrane"]
    
    def get_x_pad(self):
        return self.xypad_size
    
    def get_y_pad(self):
        return self.xypad_size
    
    def get_z_pad(self):
        return self.zpad_size
    
    def classify(self, image, x, y, z):
        #
        # Coordinates:
        #
        # Reduce the image by the padding
        # Break it into equal-sized blocks that are less than the block size
        #
        # The output image goes from <x, y, z>0 to <x, y, z>1
        # There are n_<x, y, z> blocks in each direction
        # The block coordinates are <x, y, z>s[i]:<x, y, z>s[i+1]
        #
        # The last block ends at the edge of the image.
        #
        z0 = self.get_z_pad()
        z1 = image.shape[0] - self.get_z_pad()
        n_z = 1 + int((image.shape[0] - 1) / self.block_size[0])
        zs = np.linspace(z0, z1, n_z).astype(int)
        y0 = self.get_y_pad()
        y1 = image.shape[1] - self.get_y_pad()
        n_y = 1 + int((image.shape[1] - 1) / self.block_size[1])
        ys = np.linspace(y0, y1, n_y).astype(int)
        x0 = self.get_x_pad()
        x1 = image.shape[2] - self.get_x_pad()
        n_x = 1 + int((image.shape[2] - 1) / self.block_size[2])
        xs = np.linspace(x0, x1, n_x).astype(int)
        out_image = np.zeros((z1-z0, y1 - y0, x1 - x0))
        #
        # Classify each block
        #
        for zi in range(n_z):
            if zi == n_z-1:
                z0a = image.shape[0] - self.block_size[0]
                z1a = image.shape[0]
            else:
                z0a = zs[zi] - self.get_z_pad()
                z1a = z0a + self.block_size[0]
            z0b = z0a
            z1b = z1a - self.get_z_pad() * 2
            for yi in range(n_y):
                if yi == n_y - 1:
                    y0a = image.shape[1] - self.block_size[1]
                    y1a = image.shape[1]
                else:
                    y0a = ys[yi] - self.get_y_pad()
                    y1a = y0a + self.block_size[1]
                y0b = y0a
                y1b = y1a - self.get_y_pad() * 2
                for xi in range(n_x):
                    if xi == n_x-1:
                        x0a = image.shape[2] - self.block_size[2]
                        x1a = image.shape[2]
                    else:
                        x0a = xs[xi] - self.get_x_pad()
                        x1a = x0a + self.block_size[2]
                    x0b = x0a
                    x1b = x1a - self.get_x_pad() * 2
                    block = image[z0a:z1a, y0a:y1a, x0a:x1a]
                    block.shape = [1] + list(block.shape)
                    t0 = time.time()
                    block = KerasClassifier.normalize_image(block) - .5
                    pred = self.model.predict(block)
                    logger.report_event(
                        "Processed block %d:%d, %d:%d, %d:%d in %f sec" %
                        (x0a, x1a, y0a, y1a, z0a, z1a, time.time() - t0))
                    pred.shape = (z1b - z0b, y1b - y0b, x1b - x0b)
                    out_image[z0b:z1b, y0b:y1b, x0b:x1b] = pred * 255
        return dict(membrane=out_image)
    
    @staticmethod
    def normalize_image(img, saturation_level=0.05):
        '''Normalize image to 0-1 range, removing outliers.
        
        :param img: image to be normalized
        :param saturation_level: peg values at this top and bottom percentile
        to 1 and 0 to remove outliers.
        :returns: normalized image
        '''
        sortedValues = np.sort( img.ravel())
        minVal = np.float32(sortedValues[np.int(len(sortedValues) * (saturation_level / 2))])
        maxVal = np.float32(sortedValues[np.int(len(sortedValues) * (1 - saturation_level / 2))])
        normImg = np.float32(img - minVal) * (255 / (maxVal-minVal))
        normImg[normImg<0] = 0
        normImg[normImg>255] = 255
        return (np.float32(normImg) / 255.0)

        
        