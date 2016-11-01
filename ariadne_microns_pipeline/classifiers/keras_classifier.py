'''A classifier using keras

The model is composed of the json keras model and the weights
'''
from __future__ import absolute_import

import cPickle
import hashlib
import numpy as np
import Queue
import os
from scipy.ndimage import gaussian_filter, zoom
from skimage.exposure import equalize_adapthist
import skimage
import re
import subprocess
import sys
import threading
import time

import enum
import rh_config
from rh_logger import logger
from ..targets.classifier_target import AbstractPixelClassifier

class NormalizeMethod(enum.Enum):
    '''The algorithm to use to normalize image planes'''

    '''Use a local adaptive histogram filter to normalize'''
    EQUALIZE_ADAPTHIST=1,
    '''Rescale to -.5, .5, discarding outliers'''
    RESCALE=2,
    '''Rescale 0-255 to 0-1 and otherwise do no normalization'''
    NONE=3

class KerasClassifier(AbstractPixelClassifier):
    
    has_bound_cuda = False
    models = {}

    def __init__(self, model_path, weights_path, 
                 xypad_size, zpad_size, block_size,
                 normalize_method=NormalizeMethod.EQUALIZE_ADAPTHIST,
                 downsample_factor=1.0,
                 xy_trim_size=0,
                 z_trim_size=0):
        '''Initialize from a model and weights
        
        :param model_path: path to JSON model file suitable for 
        keras.models.model_from_json
        :param weights_path: path to weights .h5 file
        :param xypad_size: padding needed in x and y
        :param zpad_size: padding needed in z
        :param block_size: size of output block to classifier
        :param normalize_method: the method to use when normalizing intensity.
              one of NormalizeMethod.EQUALIZE_ADAPTHIST or
              NormalizeMethod.RESCALE
        :param downsample_factor: Downsample the image by this scaling factor
              before classifying, then upsample by the same factor afterwards.
        :param xy_trim_size: amount to trim from edge of segmentation in the X
                            and Y direction. Trimming is done after
                            classification and before upsampling.
        :param z_trim_size: amount to trim from edge of segmentation in the Z
                           direction
        '''
        self.xypad_size = xypad_size
        self.zpad_size = zpad_size
        self.block_size = block_size
        self.model_path = model_path
        self.weights_path = weights_path
        self.model_loaded = False
        self.normalize_method = normalize_method
        self.downsample_factor = downsample_factor
        self.xy_trim_size = xy_trim_size
        self.z_trim_size = z_trim_size

    @classmethod
    def __bind_cuda(cls):
        if cls.has_bound_cuda:
            return
        import keras
        if keras.backend.backend() != 'theano':
            logger.report_event("Using Tensorflow")
            return
        t0 = time.time()
        #
        # OK - pycuda.driver.Device.count() sometimes requires
        #      pycuda.init() which sometimes screws up
        #      theano.sandbox.cuda.use. So I just use nvidia-smi to
        #      tell me about the GPUs.
        # A typical line of output:
        #      GPU 0: GeForce GTX TITAN X ...
        #
        import theano.sandbox.cuda
        nvidia_smi_output = subprocess.check_output(["nvidia-smi", "-L"])
        for line in nvidia_smi_output.split("\n"):
            match = re.search("GPU\\s(\\d+)", line)
            if match is None:
                continue
            device = int(match.group(1))
            try:
                theano.sandbox.cuda.use("gpu%d" % device, force=True)
                break
            except:
                continue
        else:
            raise RuntimeError("Failed to acquire GPU")
        logger.report_metric("gpu_acquisition_time", time.time() - t0)
        logger.report_event("Acquired GPU %d" % device)
        cls.has_bound_cuda=True
    
    def __load_model__(self):
        if self.model_loaded:
            return
        key = tuple([(path, os.stat(path).st_mtime)
                     for path in (self.model_path, self.weights_path)])
        if key in self.models:
            self.function = self.models[key]
            self.model_loaded = True
            return
        import keras
        import keras.backend as K
        from keras.optimizers import SGD
        from .cropping2d import Cropping2D
        from .depth_to_space import DepthToSpace3D
        
        if keras.backend.backend() == 'tensorflow':
            # monkey-patch tensorflow
            import tensorflow
            from tensorflow.python.ops import control_flow_ops
            tensorflow.python.control_flow_ops = control_flow_ops
        elif keras.backend.backend() == 'theano':
            import theano

        logger.report_event(
            "Loading classifier model from %s" % self.model_path)
        model_json = open(self.model_path, "r").read()
        model = keras.models.model_from_json(
            model_json,
            custom_objects={"Cropping2D":Cropping2D,
                            "DepthToSpace3D":DepthToSpace3D})
        logger.report_event(
            "Loading weights from %s" % self.weights_path)
        model.load_weights(self.weights_path)
        sgd = SGD(lr=0.01, decay=0, momentum=0.0, nesterov=False)
        logger.report_event("Compiling model")
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        if keras.backend.backend() == "theano":
            self.function = theano.function(
                model.inputs,
                model.outputs,
                givens={K.learning_phase():np.uint8(0)},
                allow_input_downcast=True,
                on_unused_input='ignore')
        else:
            self.function = model.predict
        self.models[key] = self.function
        self.model_loaded = True
        logger.report_event("Model loaded")
        
    def __getstate__(self):
        '''Get the pickleable state of the model'''
        
        return dict(xypad_size=self.xypad_size,
                    zpad_size=self.zpad_size,
                    block_size=self.block_size,
                    model_path=self.model_path,
                    weights_path=self.weights_path,
                    normalize_method=self.normalize_method.name,
                    downsample_factor=self.downsample_factor,
                    xy_trim_size=self.xy_trim_size,
                    z_trim_size=self.z_trim_size)
    
    def __setstate__(self, state):
        '''Restore the state from the pickle'''
        self.xypad_size = state["xypad_size"]
        self.zpad_size = state["zpad_size"]
        self.block_size = state["block_size"]
        self.weights_path = state["weights_path"]
        self.model_path = state["model_path"]
        self.model_loaded = False
        if "normalize_method" in state:
            self.normalize_method = NormalizeMethod[state["normalize_method"]]
        else:
            self.normalize_method = NormalizeMethod.EQUALIZE_ADAPTHIST
        if "downsample_factor" in state:
            self.downsample_factor = int(state["downsample_factor"])
        else:
            self.downsample_factor = 1
        if "xy_trim_size" in state:
            self.xy_trim_size = state["xy_trim_size"]
        else:
            self.xy_trim_size = 0
        if "z_trim_size" in state:
            self.z_trim_size = state["z_trim_size"]
    
    def get_class_names(self):
        return ["membrane"]
    
    def get_x_pad_ds(self):
        return self.xypad_size + self.xy_trim_size
    
    def get_x_pad(self):
        return self.get_x_pad_ds() / self.downsample_factor
    
    def get_y_pad_ds(self):
        return self.xypad_size + self.xy_trim_size
    
    def get_y_pad(self):
        return self.get_y_pad_ds() / self.downsample_factor
    
    def get_z_pad(self):
        return self.zpad_size + self.z_trim_size
    
    def run_via_ipc(self):
        return True
    
    def get_resources(self):
        '''Request one GPU for the classifier'''
        return dict(gpu_count=1)
    
    def classify(self, image, x, y, z):
        #
        # The threading here may seem a little odd, but Theano/CUDA want
        # to run a function on the same thread every time. So the main
        # thread runs prediction, even if it's in the middle.
        #
        t0_total = time.time()
        self.image_shape = np.array(image.shape)
        self.out_image = np.zeros((image.shape[0] - 2 * self.get_z_pad(),
                                   image.shape[1] - 2 * self.get_y_pad(),
                                   image.shape[2] - 2 * self.get_x_pad()),
                                  np.uint8)
        self.exception = None
        self.pred_queue = Queue.Queue()
        self.out_queue = Queue.Queue()
        preprocess_thread = threading.Thread(
            target=self.preprocessor,
            args=(image,))
        preprocess_thread.start()
        out_thread = threading.Thread(target=self.output_processor)
        out_thread.start()
        self.prediction_processor()
        preprocess_thread.join()
        out_thread.join()
        if self.exception is not None:
            raise self.exception
        logger.report_metric("keras_volume_classification_time",
                             time.time() - t0_total)
        return dict(membrane=self.out_image)
        
    def downsample_and_pad_image(self, image):
        '''Downsample the image according to the downsampling factor and pad it
        
        The image must be padded to at least the block size.
        
        :param image: the image at the original resolution
        
        '''
        if self.downsample_factor == 1 and \
           image.shape[1] >= self.block_size[1] and \
           image.shape[2] >= self.block_size[2]:
            return image
        image_out = []
        for plane in image:
            plane = zoom(plane, 1.0 / self.downsample_factor)
            if plane.shape[0] < self.block_size[1] or \
               plane.shape[1] < self.block_size[2]:
                xs = max(plane.shape[1], self.block_size[2])
                ys = max(plane.shape[0], self.block_size[1])
                tmp = np.zeros((ys, xs), plane.dtype)
                tmp[:plane.shape[0], :plane.shape[1]] = plane
                plane = tmp
            image_out.append(plane)
        return np.array(image_out)
    
    def preprocessor(self, image):
        '''The preprocessor thread: run normalization and make blocks'''
        import keras
        #
        # Downsample the image as a first step. All coordinates are then in
        # the downsampled size.
        #
        image = self.downsample_and_pad_image(image)
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
                      self.get_y_pad_ds()*2, 
                      self.get_x_pad_ds()*2])
        z0 = self.get_z_pad()
        z1 = image.shape[0] - self.get_z_pad()
        n_z = 1 + int((z1-z0 - 1) / output_block_size[0])
        zs = np.linspace(z0, z1, n_z+1).astype(int)
        y0 = self.get_y_pad_ds()
        y1 = image.shape[1] - self.get_y_pad_ds()
        n_y = 1 + int((y1-y0 - 1) / output_block_size[1])
        ys = np.linspace(y0, y1, n_y+1).astype(int)
        x0 = self.get_x_pad_ds()
        x1 = image.shape[2] - self.get_x_pad_ds()
        n_x = 1 + int((x1-x0 - 1) / output_block_size[2])
        xs = np.linspace(x0, x1, n_x+1).astype(int)
        t0 = time.time()
        norm_img = [
            self.normalize_image(image[zi])
            for zi in range(image.shape[0])]
        logger.report_metric("keras_cpu_block_processing_time",
                             time.time() - t0)
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
                    y0a = max(0, image.shape[1] - self.block_size[1])
                    y1a = image.shape[1]
                else:
                    y0a = ys[yi] - self.get_y_pad_ds()
                    y1a = y0a + self.block_size[1]
                y0b = y0a
                y1b = y1a - self.get_y_pad_ds() * 2
                for xi in range(n_x):
                    if xi == n_x-1:
                        x0a = max(0, image.shape[2] - self.block_size[2])
                        x1a = image.shape[2]
                    else:
                        x0a = xs[xi] - self.get_x_pad_ds()
                        x1a = x0a + self.block_size[2]
                    x0b = x0a
                    x1b = x1a - self.get_x_pad_ds() * 2
                    block = np.array([norm_img[z][y0a:y1a, x0a:x1a]
                                      for z in range(z0a, z1a)])
                    if block.shape[0] == 1:
                        if keras.backend.backend() == 'theano':
                            block.shape = \
                                [1, 1, block.shape[-2], block.shape[-1]]
                        else:
                            block.shape = \
                                [1, block.shape[-2], block.shape[-1], 1]
                    else:
                        if keras.backend.backend() == 'theano':
                            block.shape = [1, 1] + list(block.shape)
                        else:
                            block.shape = [1] + list(block.shape) + [1]
                    self.pred_queue.put((block, x0b, x1b, y0b, y1b, z0b, z1b))
        self.pred_queue.put([None] * 7)
    
    def prediction_processor(self):
        '''Run a thread to process predictions'''
        try:
            self.__bind_cuda()
            self.__load_model__()
            if not hasattr(self, "function"):
                t0 = time.time()
                self.function = cPickle.loads(self.function_pickle)
                logger.report_metric("Function load time", time.time()-t0)
                del self.function_pickle
            while True:
                block, x0b, x1b, y0b, y1b, z0b, z1b = self.pred_queue.get()
                if block is None:
                    break
                t0 = time.time()
                pred = self.function(block)[0]
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
                pred.shape = (z1b - z0b + 2 * self.z_trim_size, 
                              y1b - y0b + 2 * self.xy_trim_size,
                              x1b - x0b + 2 * self.xy_trim_size)
                pred = pred[self.z_trim_size:pred.shape[0] - self.z_trim_size,
                            self.xy_trim_size:pred.shape[1] - self.xy_trim_size,
                            self.xy_trim_size:pred.shape[2] - self.xy_trim_size]
                if self.downsample_factor != 1:
                    pred = np.array([zoom(plane, self.downsample_factor)
                                     for plane in pred])
                    y0b, y1b, x0b, x1b = \
                        [int(_ * self.downsample_factor)
                         for _ in y0b, y1b, x0b, x1b]
                # Fix padding
                if x1b > self.image_shape[2]:
                    x1b = self.image_shape[2]
                    pred = pred[:, :, :x1b - x0b]
                if y1b > self.image_shape[1]:
                    y1b = self.image_shape[1]
                    pred = pred[:, :y1b - y0b, :]
                self.out_image[z0b:z1b, y0b:y1b, x0b:x1b] = \
                    (pred * 255).astype(np.uint8)
        except:
            self.exception = sys.exc_value
            logger.report_exception()
    
    def normalize_image(self, img):
        '''Normalize an image plane's intensity to the range, -.5:.5'''
        if self.normalize_method == NormalizeMethod.EQUALIZE_ADAPTHIST:
            return self.normalize_image_adapthist(img)
        elif self.normalize_method == NormalizeMethod.RESCALE:
            return self.normalize_image_rescale(img)
        else:
            return img.astype(float) / 255.0
    
    def normalize_image_rescale(self, img, saturation_level=0.05):
        '''Normalize the image by rescaling after discaring outliers'''
        sortedValues = np.sort( img.ravel())                                        
        minVal = np.float32(
            sortedValues[np.int(len(sortedValues) * (saturation_level / 2))])                                                                      
        maxVal = np.float32(
            sortedValues[np.int(len(sortedValues) * (1 - saturation_level / 2))])                                                                  
        normImg = np.float32(img - minVal) * (255 / (maxVal-minVal))                
        normImg[normImg<0] = 0                                                      
        normImg[normImg>255] = 255                                                  
        return (np.float32(normImg) / 255.0) - .5
    
    def normalize_image_adapthist(self, img):
        '''Normalize image using a locally adaptive histogram
        
        :param img: image to be normalized
        :returns: normalized image
        '''
        version = tuple(map(int, skimage.__version__.split(".")))
        if version < (0, 12, 0):
            img = img.astype(np.uint16)
        img = equalize_adapthist(img)
        if version < (0, 12, 0):
            # Scale image if prior to 0.12
            imax = img.max()
            imin = img.min()
            img = (img.astype(np.float32) - imin) / \
                (imax - imin + np.finfo(np.float32).eps)
        return img - .5
