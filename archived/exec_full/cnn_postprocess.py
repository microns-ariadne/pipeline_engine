
import os
import sys
import time

import cv2
import numpy as np

from pipeline_common import *

from scipy.ndimage.morphology import grey_dilation

def read_probs(block_probs_path):
    
    cnn_depth_size = get_block_cnn_depth_size(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    cnn_row_size = get_block_cnn_row_size(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    cnn_col_size = get_block_cnn_col_size(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    filenames = [x for x in os.listdir(block_probs_path) if x.find('-1.png') != -1]
    filenames.sort()
    
    n_files = len(filenames)
    
    assert(n_files == cnn_depth_size)
    
    cnn_data = np.zeros((cnn_depth_size, cnn_row_size, cnn_col_size), dtype = np.uint8)
    
    for i, filename in enumerate(filenames):
        filepath = os.path.join(block_probs_path, filename)
        
        print '-- [%d] Read: %s' % (i, filepath)
        im = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        
        assert(im.shape == (cnn_row_size, cnn_col_size))
        assert(im.dtype == np.uint8)
        
        cnn_data[i,:,:] = im
    
    return (cnn_data, filenames) 

def process_cnn_data(cnn_data):
    
    print '-- process_cnn_data: start'
    
    new_cnn_data = grey_dilation(cnn_data, (2,2,2))
    
    print '-- process_cnn_data: finish'
    
    return new_cnn_data

def write_probs(
    new_cnn_data, 
    block_probs_processed_path,
    filenames):
    
    for i, filename in enumerate(filenames):
        new_im = new_cnn_data[i,:,:]
        
        filepath = os.path.join(block_probs_processed_path, filename)
        
        print '-- [%d] Write: %s' % (i, filepath)
        cv2.imwrite(filepath, new_im)
        
    
def execute(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_probs_path = get_block_probs_path(
        block_depth,
        block_row_id,
        block_col_id)
    
    block_probs_processed_path = get_block_probs_processed_path(
        block_depth,
        block_row_id,
        block_col_id)
    
    print ' -- block_probs_path           : %s' % (block_probs_path,)
    print ' -- block_probs_processed_path : %s' % (block_probs_processed_path,)
    
    verify_block_out_dir(block_probs_processed_path)
    
    (cnn_data, filenames) = read_probs(block_probs_path)
    
    new_cnn_data = process_cnn_data(cnn_data)
    
    write_probs(
        new_cnn_data, 
        block_probs_processed_path,
        filenames)
    
    return True
    
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
