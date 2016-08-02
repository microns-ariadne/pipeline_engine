'''A classifier using keras

The model is composed of the json keras model and the weights
'''

import numpy as np
import Queue
import os
from scipy.ndimage import gaussian_filter
import sys
import threading
import time

from rh_logger import logger
from ..targets.classifier_target import AbstractPixelClassifier

has_bound_cuda = False

class KerasClassifier(AbstractPixelClassifier):
    
    def __init__(self, model_path, weights_path, 
                 xypad_size, zpad_size, block_size, sigma):
        '''Initialize from a model and weights
        
        :param model_path: path to JSON model file suitable for 
        keras.models.model_from_json
        :param weights_path: path to weights .h5 file
        :param xypad_size: padding needed in x and y
        :param zpad_size: padding needed in z
        :param block_size: size of output block to classifier
        :param sigma: the standard deviation for the high-pass filter
        '''
        import keras
        self.xypad_size = xypad_size
        self.zpad_size = zpad_size
        self.block_size = block_size
        self.model_json = open(model_path, "r").read()
        self.model = keras.models.model_from_json(self.model_json)
        self.model.load_weights(weights_path)
        self.weights = self.model.get_weights()
        self.sigma = sigma
        self.__model_finished = False

    def __finish_model(self):
        '''Compile the model'''
        if self.__model_finished:
            return
        import keras
        from keras.optimizers import SGD
        t0 = time.time()
        self.model = keras.models.model_from_json(self.model_json)
        logger.report_metric("keras_model_load_time", time.time() - t0)
        t0 = time.time()
        self.model.set_weights(self.weights)
        logger.report_metric("keras_weights_load_time", time.time() - t0)
        sgd = SGD(lr=0.01, decay=0, momentum=0.0, nesterov=False)
        t0 = time.time()
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
        logger.report_metric('keras_compile_time', time.time() - t0)
        self.__model_finished = True
        
    def __getstate__(self):
        '''Get the pickleable state of the model'''
        
        return dict(xypad_size=self.xypad_size,
                    zpad_size=self.zpad_size,
                    block_size=self.block_size,
                    model=self.model_json,
                    weights=self.weights,
                    sigma=self.sigma)
    
    def __setstate__(self, state):
        '''Restore the state from the pickle'''
        self.xypad_size = state["xypad_size"]
        self.zpad_size = state["zpad_size"]
        self.block_size = state["block_size"]
        self.sigma = state["sigma"]
        self.model_json = state["model"]
        self.weights = state["weights"]
        self.__model_finished = False
    
    def get_class_names(self):
        return ["membrane"]
    
    def get_x_pad(self):
        return self.xypad_size
    
    def get_y_pad(self):
        return self.xypad_size
    
    def get_z_pad(self):
        return self.zpad_size
    
    def get_resources(self):
        '''Request one GPU for the classifier'''
        return dict(gpu_count=1)
    
    def classify(self, image, x, y, z):
        self.exception = None
        self.pred_queue = Queue.Queue()
        self.out_queue = Queue.Queue()
        pred_thread = threading.Thread(target=self.prediction_processor)
        pred_thread.start()
        out_thread = threading.Thread(target=self.output_processor)
        out_thread.start()
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
        output_block_size = self.block_size - \
            np.array([self.get_z_pad()*2, 
                      self.get_y_pad()*2, 
                      self.get_x_pad()*2])
        z0 = self.get_z_pad()
        z1 = image.shape[0] - self.get_z_pad()
        n_z = 1 + int((image.shape[0] - 1) / output_block_size[0])
        zs = np.linspace(z0, z1, n_z+1).astype(int)
        y0 = self.get_y_pad()
        y1 = image.shape[1] - self.get_y_pad()
        n_y = 1 + int((image.shape[1] - 1) / output_block_size[1])
        ys = np.linspace(y0, y1, n_y+1).astype(int)
        x0 = self.get_x_pad()
        x1 = image.shape[2] - self.get_x_pad()
        n_x = 1 + int((image.shape[2] - 1) / output_block_size[2])
        xs = np.linspace(x0, x1, n_x+1).astype(int)
        self.out_image = np.zeros((z1-z0, y1 - y0, x1 - x0), np.uint8)
        #
        # Normalize image
        #
        #temp = np.zeros(image.shape, np.float32)
        #for zi in range(image.shape[0]):
        #    temp[zi] = self.highpass(image[zi])
        #image = temp
        #del temp
        #
        # Classify each block
        #
        t0_total = time.time()
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
                    t0 = time.time()
                    if xi == n_x-1:
                        x0a = image.shape[2] - self.block_size[2]
                        x1a = image.shape[2]
                    else:
                        x0a = xs[xi] - self.get_x_pad()
                        x1a = x0a + self.block_size[2]
                    x0b = x0a
                    x1b = x1a - self.get_x_pad() * 2
                    block = KerasClassifier.normalize_image(
                        image[z0a:z1a, y0a:y1a, x0a:x1a])
                    block.shape = [1] + list(block.shape)
                    self.pred_queue.put((block, x0b, x1b, y0b, y1b, z0b, z1b))
                    logger.report_metric("keras_cpu_block_processing_time",
                                         time.time() - t0)
        self.pred_queue.put([None] * 7)
        pred_thread.join()
        out_thread.join()
        if self.exception is not None:
            raise self.exception
        logger.report_metric("keras_volume_classification_time",
                             time.time() - t0_total)
        return dict(membrane=self.out_image)
    
    def prediction_processor(self):
        '''Run a thread to process predictions'''
        try:
            global has_bound_cuda
            if not has_bound_cuda:
                has_bound_cuda=True
                import theano.sandbox.cuda
                import pycuda.driver
                for device in range(pycuda.driver.Device.count()):
                    try:
                        theano.sandbox.cuda.use("gpu%d" % device, force=True)
                        break
                    except:
                        continue
                else:
                    raise RuntimeError("Failed to acquire GPU")
                logger.report_event("Acquired GPU %d" % device)
            self.__finish_model()
            while True:
                block, x0b, x1b, y0b, y1b, z0b, z1b = self.pred_queue.get()
                if block is None:
                    break
                t0 = time.time()
                pred = self.model.predict(block)
                delta=time.time() - t0
                self.out_queue.put((pred, delta, x0b, x1b, y0b, y1b, z0b, z1b))
        except:
            self.exception = sys.exc_value
            logger.report_exception()
        self.out_queue.put([None] * 8)
    
    def output_processor(self):
        '''Run a thread to process the prediction output'''
        try:
            while True:
                pred, delta, x0b, x1b, y0b, y1b, z0b, z1b = self.out_queue.get()
                if pred is None:
                    break
                logger.report_event(
                    "Processed block %d:%d, %d:%d, %d:%d in %f sec" %
                    (x0b, x1b, y0b, y1b, z0b, z1b, delta))
                logger.report_metric("keras_block_classification_time",
                                     delta)
                pred.shape = (z1b - z0b, y1b - y0b, x1b - x0b)
                self.out_image[z0b:z1b, y0b:y1b, x0b:x1b] = \
                    (pred * 255).astype(np.uint8)
        except:
            self.exception = sys.exc_value
            logger.report_exception()
    
    def highpass(self, img):
        '''Put the image through a highpass filter to remove inconsitent bkgd
        
        Subtract the image from a Gaussian background estimator. The Gaussian
        is done in 2d - many of the artifacts are per-section.
        
        :param img: the image to be processed
        '''
        return img.astype(np.float32)
    
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

        
        
