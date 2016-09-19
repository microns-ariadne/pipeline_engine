'''Morphological operations'''

from scipy.ndimage import grey_dilation, grey_erosion

def erode_segmentation(segmentation, strel, in_place=False):
    '''Erode a segmentation using a structuring element
    
    :param segmentation: a labeling of a volume
    :param strel: the structuring element for the erosion. This should be
                  a boolean 3-d array with the voxels to erode marked as True
    :param in_place: True to erode the segmentation volume in-place, False
                     to return a new volume.
    '''
    if not in_place:
        segmentation = segmentation.copy()
    mask =\
        grey_dilation(segmentation, footprint=strel) != \
        grey_erosion(segmentation, footprint=strel)
    segmentation[mask] = 0
    return segmentation