'''Normalize image intensity for classification'''

import enum
import numpy as np
import skimage
from skimage.exposure import equalize_adapthist

class NormalizeMethod(enum.Enum):
    '''The algorithm to use to normalize image planes'''

    '''Use a local adaptive histogram filter to normalize'''
    EQUALIZE_ADAPTHIST=1,
    '''Rescale to -.5, .5, discarding outliers'''
    RESCALE=2,
    '''Rescale 0-255 to 0-1 and otherwise do no normalization'''
    NONE=3

def normalize_image_adapthist(img, offset=.5):
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
    return img - offset

def normalize_image_rescale(img, saturation_level=0.05, offset=.5):
    '''Normalize the image by rescaling after discaring outliers
    
    :param img: the image to normalize
    :param saturation_level: the fraction of outliers to discard from the
    two extrema
    :param offset: the offset to subtract from the result, scaled to 0-1
    '''
    sortedValues = np.sort( img.ravel())                                        
    minVal = np.float32(
        sortedValues[np.int(len(sortedValues) * (saturation_level / 2))])                                                                      
    maxVal = np.float32(
        sortedValues[np.int(len(sortedValues) * (1 - saturation_level / 2))])                                                                  
    normImg = np.float32(img - minVal) * (255 / (maxVal-minVal))                
    normImg[normImg<0] = 0                                                      
    normImg[normImg>255] = 255                                                  
    return (np.float32(normImg) / 255.0) - offset

def normalize_image(img, normalize_method, 
                    saturation_level=0.05,
                    offset=.5):
    '''Normalize an image plane's intensity
    
    :param img: the image to normalize
    :param normalize_method: one of the image normalization enumerations
    :param saturation_level: for the rescaling method, the fraction of outliers
    to discard from the distribution (both min and max).
    :param offset: the offset to subtract.
    '''
    if normalize_method == NormalizeMethod.EQUALIZE_ADAPTHIST:
        return normalize_image_adapthist(img, offset)
    elif normalize_method == NormalizeMethod.RESCALE:
        return normalize_image_rescale(img, saturation_level, offset)
    else:
        return img.astype(float) / 255.0


