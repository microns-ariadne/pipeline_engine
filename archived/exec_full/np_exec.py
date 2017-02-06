
import os
import sys
import time

import cv2
import numpy as np

from pipeline_common import *
    
def block_cut_padding(
    block_depth,
    block_row_id,
    block_col_id,
    block_np_path,
    Z_PAD,
    X_PAD,
    Y_PAD):
    
    filepaths = [os.path.join(block_np_path, x) for x in os.listdir(block_np_path)]
    filepaths.sort()
    
    block_size = get_block_cnn_depth_size(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_rows = get_block_cnn_row_size(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_cols = get_block_cnn_col_size(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    assert (block_size == len(filepaths)), 'block[%d,%d,%d]: unexpected block_size [%d]' % (
        block_depth,
        block_row_id,
        block_col_id,
        len(filepaths))
    
    (Z_start_idx, Z_finish_idx) = get_Z_pad_cut_indices(
        block_depth,
        block_size,
        1)
    
    (X_start_idx, X_finish_idx) = get_X_pad_cut_indices(
        block_row_id,
        block_rows)
    
    (Y_start_idx, Y_finish_idx) = get_Y_pad_cut_indices(
        block_col_id,
        block_cols)
    
    print '===================================================================='
    print '-- Reading'
    print '===================================================================='
        
    res_im_list = []
    for Z_index, filepath in enumerate(filepaths):
        print ' -- Process[%d]: %s' % (Z_index, filepath)
        im = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        
        d1 = im.shape[0]
        d2 = im.shape[1]
        
        assert (d1 == block_rows), 'block[%d,%d,%d][%d]: unexpected block_rows [%d]' % (
            block_depth,
            block_row_id,
            block_col_id,
            Z_index,
            d1)
        
        assert (d2 == block_cols), 'block[%d,%d,%d][%d]: unexpected block_cols [%d]' % (
            block_depth,
            block_row_id,
            block_col_id,
            Z_index,
            d2)
        
        if Z_index not in list(xrange(Z_start_idx, Z_finish_idx)):
            print '   -- Z_pad_skip [%d]' % (Z_index,)
            res_im_list.append((Z_index, False, filepath, None))
            continue
        
        res_im = im[ X_start_idx : X_finish_idx , Y_start_idx : Y_finish_idx ]
        
        print '   -- X/Y cut new_size: %r' % (res_im.shape,)
        
        res_im_list.append((Z_index, True, filepath, res_im))
    
    print '===================================================================='
    print '-- Writing'
    print '===================================================================='
            
    for (Z_index, is_valid, filepath, res_im) in res_im_list:
        
        if not is_valid:
            print ' -- Delete[%d]: %s' % (Z_index, filepath)
            os.unlink(filepath)
            continue
        
        print ' -- Write[%d]: %s' % (Z_index, filepath)
        im = cv2.imwrite(filepath, res_im)
        
        
def execute(
    block_depth, 
    block_row_id, 
    block_col_id,
    np_classifier_filepath):
    
    block_probs_np_path = get_block_probs_np_path(    
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_ws_path = get_block_ws_path(    
        block_depth, 
        block_row_id, 
        block_col_id)
                    
    block_np_path = get_block_np_path(    
        block_depth, 
        block_row_id, 
        block_col_id)
    
    verify_block_out_dir(block_np_path)
        
    block_np_seg_prefix = os.path.join(block_np_path, '%s_np_seg_' % (PREFIX,))
    
    np_bin_cmd = ('%s %s %s %s --threshold=%.2f --output-file=%s' % 
        (NP_BIN_PATH,
         block_ws_path,
         block_probs_np_path,
         np_classifier_filepath,
         NP_THRESHOLD,
         block_np_seg_prefix))
        
    (is_success, out_lines) = exec_cmd(np_bin_cmd)
    
    if not is_success:
        print ' -- NP failed'
        return is_success
    
    print ' -- NP success'
    
    # block_cut_padding(
    #         block_depth,
    #         block_row_id,
    #         block_col_id,
    #         block_np_path,
    #         Z_PAD,
    #         X_PAD,
    #         Y_PAD)
    #         
    return is_success
    
if '__main__' == __name__:
    try:
        (prog_name, 
         block_depth, 
         block_row_id, 
         block_col_id,
         np_classifier_filepath) = sys.argv[:5]
        
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
        block_col_id,
        np_classifier_filepath)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ELAPSED TIME: %s seconds' % (elapsed_time_secs,)
    
    if (is_success):
        print PROC_SUCCESS_STR
