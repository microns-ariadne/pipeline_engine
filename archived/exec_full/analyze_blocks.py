
import os
import sys
import time

import cv2
import numpy as np

from pipeline_common import *

MAX_ZERO_RATIO = 0.8

BORDER_MASKS_BIN_PATH = '/home/armafire/Pipeline/pipeline_engine/matlab_scripts/run_border_masks.sh'
BORDER_WIDTH = 256
CLOSE_WIDTH = 26

def get_processed_im(im_id, im_type, orig_block_images, meta_block_images):
    orig_im = orig_block_images[im_id]
    meta_im = meta_block_images[im_id]
    
    if im_type == TYPE_COMPLEX:
        res_im = orig_im
        
    else:
        mask_im = meta_im != 255
        res_im = orig_im * mask_im
    
    return res_im
    

def get_filenames(input_dir):
    filenames = os.listdir(input_dir)
    
    res_filenames = []
    for filename in filenames:
        filepath = os.path.join(input_dir, filename)
        
        if os.path.isdir(filepath):
            continue
        
        if filename.find('.txt') != -1:
            continue
        
        res_filenames.append(filename)
    
    res_filenames.sort()
    
    return res_filenames
    
def read_block_images(block_em_path):
    filepaths = [os.path.join(block_em_path, x) for x in os.listdir(block_em_path)]
    filepaths.sort()
    
    block_images = []
    for i, filepath in enumerate(filepaths):
        print '  -- read_block_images[%d]: %s' % (i, filepath)
        im = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise Exception('ERROR: read_block_images failed for %s' % (filepath,))
        block_images.append(im)
        
    return block_images
    
def is_block_image_valid(block_image):
    
    d1 = block_image.shape[0]
    d2 = block_image.shape[1]
    n_pixels = d1 * d2
    
    zero_mask = (block_image == 0)
    n_zeros = zero_mask.sum()
    
    return ((float(n_zeros) / float(n_pixels)) < MAX_ZERO_RATIO)
    
def execute(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_em_path = get_block_em_path(
        block_depth,
        block_row_id,
        block_col_id)
        
    block_meta_path = get_block_meta_path(
        block_depth,
        block_row_id,
        block_col_id)
    
    print ' -- block_em_path   : %s' % (block_em_path,)
    print ' -- block_meta_path : %s' % (block_meta_path,)
    
    if not os.path.exists(block_meta_path):
        print ' -- [%.4d-%.4d-%.4d] creating meta-dir: %s' % (
            block_depth, 
            block_row_id, 
            block_col_id,
            block_meta_path)
        os.makedirs(block_meta_path)
    
    print ' -- verify that all images are above the zero threshold'
    
    block_images = read_block_images(block_em_path)   
    
    for i, block_image in enumerate(block_images):
        if not is_block_image_valid(block_image):
            print '  -- image %d is not valid' % (i,)
            meta_update_block_status_file(
                block_depth,
                block_row_id,
                block_col_id,
                META_BLOCK_STATUS_NOT_VALID)
            return True
    
    print '  -- all images are valid'
    meta_update_block_status_file(
        block_depth,
        block_row_id,
        block_col_id,
        META_BLOCK_STATUS_VALID)
    # Generate border meta files
    
    print ' -- generate border meta files'
    border_masks_cmd = '%s %d %d %s %s' % (
        BORDER_MASKS_BIN_PATH,
        BORDER_WIDTH,
        CLOSE_WIDTH,
        block_em_path,
        block_meta_path)
        
    (is_success, out_lines) = exec_cmd(border_masks_cmd)
    
    print ' -- done'
    
    return is_success
    
if '__main__' == __name__:
    try:
        (prog_name, 
         block_depth, 
         block_row_id, 
         block_col_id) = sys.argv[:4]
        
        block_depth = int(block_depth)
        block_row_id = int(block_row_id)
        block_col_id = int(block_col_id)
        
    except ValueError, e:
        sys.exit('USAGE: %s \
    [block_depth] \
    [block_row_id] \
    [block_col_id]' % (sys.argv[0],))
    
    start_time_secs = time.time()
    
    is_success = execute(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ELAPSED TIME: %s seconds' % (elapsed_time_secs,)
    
    if (is_success):
        print PROC_SUCCESS_STR
