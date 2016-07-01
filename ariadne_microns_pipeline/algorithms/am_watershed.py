import _am_watershed
import enum
import numpy as np

class Connectivity(enum.Enum):
    '''Two-dimensional 4-connectivity'''
    FOUR=4,
    '''Three-dimensional 6-connectivity'''
    SIX=6
def watershed(image, markers, connectivity=Connectivity.SIX, output=False):
    '''Perform a seeded watershed

    :param image: 3 dimensional uint8 image being watershedded
    :param markers: array of dimensions similar to image where positive values
        mark the watershed seeds. 
    :param connectivity: either Connectivity.FOUR for 2D or Connectivity.SIX
        for 3d.
    :param output: if True, the watershed is done in-place and the markers
        array is overwritten with the segmentation. In this case,
        the markers array must be a contiguous array with dtype of uint32.
    :returns: the markers array.
    '''
    if not output:
        temp = np.zeros(markers.shape, np.uint32)
        temp[:] = markers[:]
        markers = temp
    _am_watershed.watershed(
        np.ascontiguousarray(image, np.uint8), markers, connectivity.value)
    return markers

