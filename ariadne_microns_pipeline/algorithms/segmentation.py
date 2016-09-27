'''Segmentation algorithms and heuristics'''

import numpy as np
from scipy.ndimage import gaussian_filter, label, find_objects

def segment_vesicle_style(prob,
                         sigma_xy,
                         sigma_z,
                         threshold,
                         min_size_2d,
                         max_size_2d,
                         min_size_3d,
                         min_slice):
    '''Segment according to the "Vesicle" algorithm

    See http://arxiv.org/abs/1403.3724
    VESICLE: Volumetric Evaluation of Synaptic Interfaces using Computer
             vision at Large Scale
    William Gray Roncal, Michael Pekala, Verena Kaynig-Fittkau, 
    Dean M. Kleissas, Joshua T. Vogelstein, Hanspeter Pfister, Randal Burns, 
    R. Jacob Vogelstein, Mark A. Chevillet, Gregory D. Hager

    :param prob: The synapse probability volume with axes of z, y, and x
    :param sigma_xy: The sigma for the smoothing gaussian in the x and y
                     directions
    :param sigma_z: The sigma for the smoothing Gaussian in the z direction
    :param threshold: The probability threshold above which, a voxel is
                      deemed to be part of a synapse.
    :param min_size_2d: discard any 2d segments with area less than this.
    :param max_size_2d: discard any 2d segments with area greater than this.
    :param min_size_3d: discard any 3d segments with area less than this.
    :param min_slice: discard any 3d segments whose z-extent is less than this.
    '''

    fg = gaussian_filter(prob.astype(np.float32),
                         (sigma_z, sigma_xy, sigma_xy)) > threshold
    #
    # 2D filter by area
    #
    strel = np.array([[False, True, False],
                      [ True, True, True],
                      [False, True, False]])
    for z, plane in enumerate(fg):
        l, count = label(plane)
        areas = np.bincount(l.flatten())
        bad = np.zeros(areas.shape[0], bool)
        bad[areas < min_size_2d] = True
        bad[areas > max_size_2d] = True
        fg[z, bad[l]] = False
    #
    # 3D filter by area
    #
    strel = np.array([[[False, False, False],
                       [False, True, False],
                       [False, False, False]],
                      [[False, True, False],
                       [True,  True, True],
                       [False, True, False]],
                      [[False, False, False],
                       [False, True, False],
                       [False, False, False]]])
    l, count = label(fg, strel)
    areas = np.bincount(l.flatten())
    bad = areas < min_size_3d
    #
    # 3D filter by z
    #
    for i, location in enumerate(find_objects(l)):
        if location is None:
            bad[i+1] = True
        z = location[0]
        if z.stop - z.start < min_slice:
            bad[i+1] = True
    fg[bad[l]] = False
    #
    # Do 18-connected connected-components
    #
    strel = np.ones((3, 3, 3), bool)
    for z in 0, -1:
        for y in 0, -1:
            for x in 0, -1:
                strel[z, y, x] = False
    l, count = label(fg, strel)
    return l

