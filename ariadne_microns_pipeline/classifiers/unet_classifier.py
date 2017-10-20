'''The affinity UNET model, in Donlai's fork of Keras.

To run:

Start IPC workers using an environment that contains Theano 0.10 and
Keras from here: https://github.com/donglaiw/keras.
'''
import rh_logger
import cPickle
from .keras_classifier import KerasClassifier
from ..algorithms.normalize import NormalizeMethod, normalize_image

class UnetClassifier(KerasClassifier):
    
    def __init__(self,
                 weights_path,
                 classes,
                 xypad_size,
                 zpad_size,
                 block_size,
                 normalize_method,
                 normalize_offset=None,
                 normalize_saturation_level=None,
                 downsample_factor=1.0):
        '''Initialize the UNET model with given weights
        
        :param weights_path: the path to the HDF5 weights file
        :param classes: the names of the classes in the order they appear
        in the output tensor.
        :param xypad_size: the input padding required in the X and Y direction
        :param zpad_size: the input padding required in the Z direction
        :param block_size: the Z, Y, X dimensions of the network's
        output
        :param normalize_method: One of the NormalizeMethod enums - how
        to normalize the input image
        :param  normalize_offset: The offset to subtract from the input
        image after normalization
        :param normalize_saturation_level: for the RESCALE method, the
        percent of outlier pixels to pin at 0 and 1 before rescaling
        :param downsample_factor: The number of pixels in the original image
        to downsample to a single pixel in the image to be fed to the
        neural net. For instance, downsample_factor=2.0 means make a
        204 x 204 x 31 image out of a 408 x 408 x 31 image.
        '''
        self.weights_path = weights_path
        self.classes = classes
        self.xypad_size = xypad_size
        self.zpad_size = zpad_size
        self.block_size = block_size
        self.normalize_method = normalize_method
        self.normalize_offset = normalize_offset
        self.normalize_saturation_level = normalize_saturation_level
        self.downsample_factor = downsample_factor
        self.model_loaded = False
        self.model_path = None
    
    def __getstate__(self):
        return dict(weights_path=self.weights_path,
                    classes = self.classes,
                    xypad_size=self.xypad_size,
                    zpad_size=self.zpad_size,
                    block_size=self.block_size,
                    normalize_method=self.normalize_method,
                    normalize_offset=self.normalize_offset,
                    normalize_saturation_level=self.normalize_saturation_level,
                    downsample_factor=self.downsample_factor)
    
    def __setstate__(self, state):
        self.model_path = None
        self.model_loaded = False
        self.weights_path = state["weights_path"]
        self.classes = state["classes"]
        self.xypad_size = state["xypad_size"]
        self.zpad_size = state["zpad_size"]
        self.block_size = state["block_size"]
        self.normalize_method = state["normalize_method"]
        self.normalize_offset = state["normalize_offset"]
        self.normalize_saturation_level = state["normalize_saturation_level"]
        self.downsample_factor = state["downsample_factor"]
        
    def _load_model(self):
        if self.model_loaded:
            return
        import keras
        from keras.applications.affinity_unet import get_unet, load_weights
        
        self.model = get_unet()
        with open(self.weights_path, "rb") as fd:
            weights = cPickle.load(fd)
        load_weights(self.model, weights)

    