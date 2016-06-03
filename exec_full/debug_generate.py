
import sys
import os
import time
import shutil

import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import h5py
import cilk_rw

from scipy.ndimage.morphology import grey_dilation

from pipeline_common import *

IS_GENERATE_EM = 0

#IS_GENERATE_PROBS = 0

IS_GENERATE_PROBS_WS = 0

IS_GENERATE_PROBS_NP = 0
 
IS_GENERATE_WS = 0
 
IS_GENERATE_NP = 0

IS_GENERATE_MERGE = 1

def fix_merge_Z_overlap(
    block_depth_id,
    Z_start_idx, 
    Z_finish_idx,
    MERGE_PREPROCESS_Z_OVERLAP):
    
    start_idx = Z_start_idx
    finish_idx = Z_finish_idx
    
    z_pad_left = MERGE_PREPROCESS_Z_OVERLAP
    z_pad_right = MERGE_PREPROCESS_Z_OVERLAP
                
    if block_depth > BLOCKS_MIN_DEPTH:
        start_idx += z_pad_left
    
    if block_depth < BLOCKS_MAX_DEPTH:
        finish_idx -= z_pad_right
    
    return (start_idx, finish_idx)

def get_real_data_indices(
    block_depth_id,
    block_row_id,
    block_col_id,
    Z_start_idx,
    Z_finish_idx,
    X_start_idx,
    X_finish_idx,
    Y_start_idx,
    Y_finish_idx):
    
    data_Z_start_idx = 0
    data_Z_finish_idx = BLOCK_N_IMAGES
    
    Z_len = Z_finish_idx - Z_start_idx
    assert(Z_len <= data_Z_finish_idx), 'unexpected Z_len = %d' % (Z_len,)
    if Z_len < data_Z_finish_idx:
        if block_depth_id == BLOCKS_MIN_DEPTH:
            data_Z_start_idx += CNN_DEPTH_LEG
            
        if block_depth_id == BLOCKS_MAX_DEPTH:
            data_Z_finish_idx -= CNN_DEPTH_LEG
    
    data_X_start_idx = 0
    data_X_finish_idx = BLOCK_N_ROWS / BLOCK_SUB_SAMPLE
    
    X_len = X_finish_idx - X_start_idx
    assert(X_len <= data_X_finish_idx), 'unexpected X_len = %d' % (X_len,)
    if X_len < data_X_finish_idx:
        if block_row_id == BLOCKS_MIN_ROW_ID:
            data_X_start_idx += CNN_PATCH_LEG
            
        if block_row_id == BLOCKS_MAX_ROW_ID:
            data_X_finish_idx -= CNN_PATCH_LEG
    
    data_Y_start_idx = 0
    data_Y_finish_idx = BLOCK_N_COLS / BLOCK_SUB_SAMPLE
    
    Y_len = Y_finish_idx - Y_start_idx
    assert(Y_len <= data_Y_finish_idx), 'unexpected Y_len = %d' % (Y_len,)
    if Y_len < data_Y_finish_idx:
        if block_col_id == BLOCKS_MIN_COL_ID:
            data_Y_start_idx += CNN_PATCH_LEG
            
        if block_col_id == BLOCKS_MAX_COL_ID:
            data_Y_finish_idx -= CNN_PATCH_LEG
            
    return ((data_Z_start_idx, data_Z_finish_idx), 
            (data_X_start_idx, data_X_finish_idx), 
            (data_Y_start_idx, data_Y_finish_idx))
    
def normalize_3D_block_uint32(block_data):
    block_data = block_data.astype(np.uint64)
    block_data *= 1000000
    block_data %= 16777213
    block_data = block_data.astype(np.uint32)
    return block_data
    
def get_RGB(im):
    
    d1 = im.shape[0]
    d2 = im.shape[1]
    
    im_diff_x = np.diff(im, axis = 0)
    im_diff_y = np.diff(im, axis = 1)
    
    mask = np.zeros((d1,d2))
    mask[0:-1,:] += im_diff_x
    mask[:,0:-1] += im_diff_y
    
    mask_1 = (mask > 0).astype(np.uint32)
    mask_2 = (mask == 0).astype(np.uint32)
    
    mask_1 = grey_dilation(mask_1, size = (3,3))
    mask_1 = (mask_1 == 0).astype(np.uint32)
    mask_2 = (mask_2 * mask_1).astype(np.uint32)
    
    im_new = np.zeros((d1,d2), dtype = np.uint32)
    #im_new += (mask_1 * (int(math.pow(2,24)) - 1))
    im_new += (im * mask_2)
    
    #cmap = plt.get_cmap('hsv')
    #im_rgba = (cmap(im_new.astype(np.float) / im_new.max()) * 255).astype(np.uint8)
    
    im_rgb = np.zeros((d1,d2,3), dtype = np.uint8)

    im_rgb[:,:,0] = (im_new) % 256
    im_rgb[:,:,1] = ((im_new) / 256) % 256
    im_rgb[:,:,2] = ((im_new) / 256 / 256) % 256
            
    return im_rgb

def debug_generate_from_block(
    block_depth_id, 
    block_row_id, 
    block_col_id,
    block_path,
    block_debug_path,
    postfix,
    is_RGB,
    is_padded,
    is_EM):

    print 'debug_generate_from_block: start'
    
    block_size = get_block_cnn_depth_size(
        block_depth_id, 
        block_row_id, 
        block_col_id)

    block_rows = get_block_cnn_row_size(
        block_depth_id, 
        block_row_id, 
        block_col_id)

    block_cols = get_block_cnn_col_size(
        block_depth_id, 
        block_row_id, 
        block_col_id)
    
    print '  -- block_size: [%d,%d,%d]' % (
        block_size, 
        block_rows, 
        block_cols)
    
    if is_RGB:
        block_data = cilk_rw.read_rgb_labels(block_path)
    else:
        block_data = cilk_rw.read_probabilities_int(block_path)
    
    if is_EM:
        assert(is_RGB == False)
        
        block_data_cnn_resized = np.zeros((
            block_size, 
            block_rows, 
            block_cols), dtype = np.uint8)
        
        for i in xrange(1, block_data.shape[0] - 1):
            im = block_data[i,:,:]
            im = im[ CNN_PATCH_LEG : -CNN_PATCH_LEG, 
                     CNN_PATCH_LEG : -CNN_PATCH_LEG]
            im = cv2.resize(
                im, 
                None, 
                fx = 0.5, 
                fy = 0.5, 
                interpolation = cv2.INTER_NEAREST)
            
            block_data_cnn_resized[i - 1, :, :] = im
        
        block_data = block_data_cnn_resized
    
    n_files = block_data.shape[0]
    n_rows = block_data.shape[1]
    n_cols = block_data.shape[2]
    
    if is_padded:
        assert (n_files == block_size), 'block[%d,%d,%d]: unexpected block_size [%d]' % (
            block_depth_id,
            block_row_id,
            block_col_id,
            n_files)

        assert (n_rows == block_rows), 'block[%d,%d,%d][%d]: unexpected block_rows [%d]' % (
            block_depth_id,
            block_row_id,
            block_col_id,
            Z_index,
            n_rows)

        assert (n_cols == block_cols), 'block[%d,%d,%d][%d]: unexpected block_cols [%d]' % (
            block_depth_id,
            block_row_id,
            block_col_id,
            Z_index,
            n_cols)
    
    (Z_start_idx, Z_finish_idx) = get_Z_pad_cut_indices(
        block_depth_id,
        block_size,
        z_pad_right_overlap = 0)
                
    (Z_start_idx, Z_finish_idx) = (0, n_files)        
    (X_start_idx, X_finish_idx) = (0, n_rows)
    (Y_start_idx, Y_finish_idx) = (0, n_cols)
    
    if MERGE_PREPROCESS_Z_OVERLAP > 0:
        (Z_start_idx, Z_finish_idx) = fix_merge_Z_overlap(
            block_depth_id,
            Z_start_idx, 
            Z_finish_idx,
            MERGE_PREPROCESS_Z_OVERLAP)
    
    if is_padded:    
        (Z_start_idx, Z_finish_idx) = get_Z_pad_cut_indices(
            block_depth_id,
            block_size,
            z_pad_right_overlap = 0)

        (X_start_idx, X_finish_idx) = get_X_pad_cut_indices(
            block_row_id,
            block_rows)

        (Y_start_idx, Y_finish_idx) = get_Y_pad_cut_indices(
            block_col_id,
            block_cols)
    
    print 'Z/X/Y cuts = [%d-%d][%d-%d][%d-%d]' % (
        Z_start_idx, Z_finish_idx,
        X_start_idx, X_finish_idx,
        Y_start_idx, Y_finish_idx)
    
    verify_block_out_dir(block_debug_path)
    
    cur_dtype = np.uint8
    if is_RGB:
        block_data = normalize_3D_block_uint32(block_data)
        cur_dtype = np.uint32
    
    real_block_data = np.zeros((
        BLOCK_N_IMAGES, 
        BLOCK_N_ROWS / BLOCK_SUB_SAMPLE, 
        BLOCK_N_COLS / BLOCK_SUB_SAMPLE), dtype = cur_dtype)
    
    ((real_Z_start_idx, real_Z_finish_idx),
     (real_X_start_idx, real_X_finish_idx),
     (real_Y_start_idx, real_Y_finish_idx)) = get_real_data_indices(
        block_depth_id,
        block_row_id,
        block_col_id,
        Z_start_idx,
        Z_finish_idx,
        X_start_idx,
        X_finish_idx,
        Y_start_idx,
        Y_finish_idx)
    
    print 'Z/X/Y real = [%d-%d][%d-%d][%d-%d]' % (
        real_Z_start_idx, real_Z_finish_idx,
        real_X_start_idx, real_X_finish_idx,
        real_Y_start_idx, real_Y_finish_idx)
    
    real_block_data[ 
        real_Z_start_idx : real_Z_finish_idx , 
        real_X_start_idx : real_X_finish_idx , 
        real_Y_start_idx : real_Y_finish_idx] = block_data[
        Z_start_idx : Z_finish_idx , 
        X_start_idx : X_finish_idx , 
        Y_start_idx : Y_finish_idx ]
    
    for cur_depth in xrange(real_block_data.shape[0]):
        im = real_block_data[cur_depth, :, :]
        
        if is_RGB:
            im = get_RGB(im)
        
        out_filename = '%s_block_%.4d_%.4d_%.4d_%.4d_%s.png' % (
            PREFIX,
            block_depth,
            block_row,
            block_col,
            cur_depth,
            postfix)
            
        out_filepath = os.path.join(block_debug_path, out_filename)
        
        print '  -- Write[%d]: %s' % (cur_depth, out_filepath)
        
        cv2.imwrite(out_filepath, im)
    
    print 'debug_generate_from_block: finish'

def debug_generate_em(
    block_depth_id, 
    block_row_id, 
    block_col_id):

    print 'debug_generate_probs_em: start'
    
    block_em_path = get_block_em_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
    
    block_debug_path = get_block_debug_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
        
    block_debug_em_path = os.path.join(block_debug_path, EM_DIR)
    
    postfix = 'em'
    
    debug_generate_from_block(
        block_depth_id, 
        block_row_id, 
        block_col_id,
        block_em_path,
        block_debug_em_path,
        postfix,
        is_RGB = False,
        is_padded = True,
        is_EM = True)
    
    print 'debug_generate_em: finish'

def debug_generate_probs_ws(
    block_depth_id, 
    block_row_id, 
    block_col_id):

    print 'debug_generate_probs_ws: start'
    
    block_probs_ws_path = get_block_probs_ws_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
    
    block_debug_path = get_block_debug_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
        
    block_debug_probs_ws_path = os.path.join(block_debug_path, PROBS_WS_DIR)
    
    postfix = 'probs_ws'
    
    debug_generate_from_block(
        block_depth_id, 
        block_row_id, 
        block_col_id,
        block_probs_ws_path,
        block_debug_probs_ws_path,
        postfix,
        is_RGB = False,
        is_padded = True,
        is_EM = False)
    
    print 'debug_generate_probs_ws: finish'

def debug_generate_probs_np(
    block_depth_id, 
    block_row_id, 
    block_col_id):

    print 'debug_generate_probs_np: start'
    
    block_probs_np_path = get_block_probs_np_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
    
    block_debug_path = get_block_debug_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
        
    block_debug_probs_np_path = os.path.join(block_debug_path, PROBS_NP_DIR)
    
    postfix = 'probs_np'
    
    debug_generate_from_block(
        block_depth_id, 
        block_row_id, 
        block_col_id,
        block_probs_np_path,
        block_debug_probs_np_path,
        postfix,
        is_RGB = False,
        is_padded = True,
        is_EM = False)
    
    print 'debug_generate_probs_np: finish'

def debug_generate_ws(
    block_depth_id, 
    block_row_id, 
    block_col_id):

    print 'debug_generate_ws: start'
    
    block_ws_path = get_block_ws_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
    
    block_debug_path = get_block_debug_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
        
    block_debug_ws_path = os.path.join(block_debug_path, WS_DIR)
    
    postfix = 'ws'
    
    debug_generate_from_block(
        block_depth_id, 
        block_row_id, 
        block_col_id,
        block_ws_path,
        block_debug_ws_path,
        postfix,
        is_RGB = True,
        is_padded = True,
        is_EM = False)
    
    print 'debug_generate_ws: finish'

def debug_generate_np(
    block_depth_id, 
    block_row_id, 
    block_col_id):

    print 'debug_generate_np: start'
    
    block_np_path = get_block_np_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
    
    block_debug_path = get_block_debug_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
        
    block_debug_np_path = os.path.join(block_debug_path, NP_DIR)
    
    postfix = 'np'
    
    debug_generate_from_block(
        block_depth_id, 
        block_row_id, 
        block_col_id,
        block_np_path,
        block_debug_np_path,
        postfix,
        is_RGB = True,
        is_padded = True,
        is_EM = False)
    
    print 'debug_generate_np: finish'

def debug_generate_merge(
    block_depth_id, 
    block_row_id, 
    block_col_id):

    print 'debug_generate_merge: start'
    
    block_merge_path = get_block_merge_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
        
    block_merge_out_path = os.path.join(
        block_merge_path, 
        'out_segmentation_%.4d_%.4d_%.4d' % (
            block_depth_id, 
            block_row_id, 
            block_col_id))
    
    block_debug_path = get_block_debug_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
        
    block_debug_merge_path = os.path.join(block_debug_path, MERGE_DIR)
    
    postfix = 'merge'
    
    debug_generate_from_block(
        block_depth_id, 
        block_row_id, 
        block_col_id,
        block_merge_out_path,
        block_debug_merge_path,
        postfix,
        is_RGB = True,
        is_padded = False,
        is_EM = False)
    
    print 'debug_generate_merge: finish'

def execute(
    block_depth_id,
    block_row_id,
    block_col_id):
    
    if not meta_is_block_valid(
        block_depth_id, 
        block_row_id, 
        block_col_id):
        print ' -- %s is not valid [SKIP]' % (block_name,)
        return
    
    block_debug_path = get_block_debug_path(    
        block_depth_id, 
        block_row_id, 
        block_col_id)
    
    if not os.path.exists(block_debug_path):
        os.makedirs(block_debug_path)
        
    if IS_GENERATE_EM:
        debug_generate_em(
            block_depth_id,
            block_row_id,
            block_col_id)
    
    # if IS_GENERATE_PROBS:
    #         debug_generate_probs(
    #             block_depth_id,
    #             block_row_id,
    #             block_col_id)
    
    if IS_GENERATE_PROBS_WS:
        debug_generate_probs_ws(
            block_depth_id,
            block_row_id,
            block_col_id)
    
    if IS_GENERATE_PROBS_NP:
        debug_generate_probs_np(
            block_depth_id,
            block_row_id,
            block_col_id)
    
    if IS_GENERATE_WS:   
        debug_generate_ws(
            block_depth_id,
            block_row_id,
            block_col_id)     
    
    if IS_GENERATE_NP:
        debug_generate_np(
            block_depth_id,
            block_row_id,
            block_col_id)
    
    if IS_GENERATE_MERGE:
        debug_generate_merge(
            block_depth_id,
            block_row_id,
            block_col_id)
        
    

if '__main__' == __name__:
    try:
        (prog_name, 
         block_depth,
         block_row,
         block_col) = sys.argv[:4]
        
        block_depth = int(block_depth)
        block_row = int(block_row)
        block_col = int(block_col)
        
    except ValueError, e:
        sys.exit('USAGE: %s \
            [block_depth] \
            [block_row] \
            [block_col]' % (sys.argv[0],))

    execute(
        block_depth,
        block_row,
        block_col)
    
    print PROC_SUCCESS_STR
    
