#!/usr/bin/env python

import sys
import os
import os.path
import numpy
import shutil
import logging
import json
from skimage import morphology as skmorph
from scipy.ndimage import label
import traceback
import glob
import re
import h5py
import hashlib

# don't need pixel?
import imio, morpho, app_logger, session_manager

# Group where we store predictions in HDF5 file
PREDICTIONS_HDF5_GROUP = '/volume/predictions'


def grab_boundary(prediction, channels, master_logger):
    boundary = None
    master_logger.debug("Grabbing boundary labels: " + str(channels))
    for channel_id in channels:
        if boundary is None:
            boundary = prediction[...,channel_id] 
        else:
            boundary += prediction[...,channel_id]

    return boundary


def gen_supervoxels(options, prediction_file, master_logger): # border_size=0.0, seed_val=0.0):
    """Returns ndarray labeled using (optionally seeded) watershed algorithm

    Args:
        options:  OptionNamespace.
        prediction_file:  String.  File name of prediction hdf5 file where predictions
            are assumed to be in group PREDICTIONS_HDF5_GROUP.

    Returns:
        A 2-tuple of supervoxel and prediction ndarray.
    """
    master_logger.debug("Generating supervoxels")
    if not os.path.isfile(prediction_file):
        raise Exception("Training file not found: " + prediction_file)

    prediction = imio.read_image_stack(prediction_file, group=PREDICTIONS_HDF5_GROUP)
    master_logger.info("Shape of prediction: %s" % str(prediction.shape))
    master_logger.info("Transposed boundary prediction")
    # if prediction.ndim == 3:
    #     prediction = numpy.array([prediction, prediction])
    #     prediction = prediction.transpose((1, 2, 3, 0))
    #     print prediction.shape
    prediction = prediction.transpose((2, 1, 0, 3))

    #if options.extract_ilp_prediction:
    # prediction = 1 - prediction/255.0
    # print prediction
    # prediction = prediction.transpose((2, 1, 0))

    # TODO -- Refactor.  If 'single-channel' and hdf5 prediction file is given, it looks like
    #   read_image_stack will return a modified volume and the bound-channels parameter must
    #   be 0 or there'll be conflict.
    boundary = grab_boundary(prediction, options.bound_channels, master_logger) 
    # boundary = prediction[...] 
    master_logger.info("Shape of boundary: %s" % str(boundary.shape))

    # Prediction file is in format (t, x, y, z, c) but needs to be in format (z, x, y).
    # Also, raveler convention is (0,0) sits in bottom left while ilastik convention is
    # origin sits in top left.
    # imio.read_image_stack squeezes out the first dim.

    master_logger.info("watershed seed value threshold: " + str(options.seed_val))
    seeds = label(boundary<=options.seed_val)[0]
    print "SEEDS", (repr(seeds))
    print "LENGTH OF SEEDS: ", len(seeds)

    if options.seed_size > 0:
        master_logger.debug("Removing small seeds")
        seeds = morpho.remove_small_connected_components(seeds, options.seed_size)
        master_logger.debug("Finished removing small seeds")

    master_logger.info("Starting watershed")
    
    boundary_cropped = boundary
    seeds_cropped = seeds 
    if options.border_size > 0:
        boundary_cropped = boundary[options.border_size:(-1*options.border_size), options.border_size:(-1*options.border_size),options.border_size:(-1*options.border_size)]
        seeds_cropped = label(boundary_cropped<=options.seed_val)[0]
        if options.seed_size > 0:
            seeds_cropped = morpho.remove_small_connected_components(seeds_cropped, options.seed_size)

    # Returns a matrix labeled using seeded watershed
    watershed_mask = numpy.ones(boundary_cropped.shape).astype(numpy.uint8)
    
    # Used to specify region to ignore
    masked_bboxes = []

    if options.mask_file is not None:
        mask_file = open(options.mask_file)
        for line in mask_file:
            br = line.split()
            if len(br) == 6:
                watershed_mask[int(br[2]):(int(br[5])+1),
                            int(br[1]):(int(br[4])+1),int(br[0]):(int(br[3])+1)] = 0
                masked_bboxes.append(br)
        mask_file.close()

    supervoxels_cropped = skmorph.watershed(boundary_cropped, seeds_cropped, None, None, watershed_mask)
    
    supervoxels = supervoxels_cropped
    if options.border_size > 0:
        supervoxels = seeds.copy()
        supervoxels.dtype = supervoxels_cropped.dtype
        supervoxels[:,:,:] = 0 
        supervoxels[options.border_size:(-1*options.border_size), 
                options.border_size:(-1*options.border_size),options.border_size:(-1*options.border_size)] = supervoxels_cropped

    master_logger.info("Finished watershed")

    return supervoxels, prediction


# do these have to be h5s or tiffs?
def run_segmentation_pipeline(image_stack_filename, prob_stack_filename):
    """Runs segmentation pipeline given classifier and input image in options.

    Args:
        session_location:  String.  Export data location.
        options:  OptionNamespace.  Basically a dict with keys corresponding
            to slightly altered names ('_' instead of '-') within JSON config file.

    Returns:
        A 2-tuple of supervoxel and prediction ndarray.
    """
    # read grayscale
    if image_stack_filename is None:
        raise Exception("Must specify path to grayscale in 'image-stack'")

    print "Gen Pixel: No"
    prediction_file  = prob_stack_filename #options.pixelprob_file

    # generate supervoxels -- produces supervoxels and output as appropriate
    supervoxels = None
    prediction = None
    print "Generating supervoxels:"
    supervoxels, prediction = gen_supervoxels(prediction_file)

    # write superpixels out to hdf5 and/or raveler files
    sps_out = None
    image_stack = None

    print "Before None check"
    if supervoxels is not None:
	print "Is not None"
        print "Going to write to h5 format:"
        print "Location: supervoxels.h5"
        imio.write_image_stack(supervoxels,
                               # session_location + "/" + options.supervoxels_name, compression='lzf')
                               "supervoxels.h5", compression=None)
        print "Compression: None"


'''    options_parser.create_option("supervoxels-name", "Name for the supervoxel segmentation", 
        default_val="supervoxels.h5", required=False, dtype=str, verify_fn=None, num_args=None,
        shortcut=None, warning=False, hidden=True) 

    options_parser.create_option("supervoxels-file", "Supervoxel segmentation file or directory stack", 
        default_val=None, required=False, dtype=str, verify_fn=supervoxels_file_verify, num_args=None,
        shortcut=None, warning=False, hidden=True) 
   
    options_parser.create_option("gen-supervoxels", "Enable supervoxel generation", 
        default_val=False, required=False, dtype=bool, verify_fn=gen_supervoxels_verify, num_args=None,
        shortcut='GS', warning=True, hidden=False) 
'''

if __name__ == "__main__":
    print sys.argv
    em_stack = sys.argv[1]
    prob_stack = sys.argv[2]
    run_segmentation_pipeline(em_stack, prob_stack)
