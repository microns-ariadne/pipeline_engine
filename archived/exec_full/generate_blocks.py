import os
import sys
import time

import cv2
import numpy as np

from pipeline_common import *

def read_input(file_id):
    
    in_filename = ALIGN_RES_FILE_TEMPLATE % (file_id,)
    in_data_dir_path = get_align_result_dir(file_id)
    in_filepath = os.path.join(in_data_dir_path, in_filename)
    
    print 'read_input[%d]: %s' % (file_id, in_filepath)
    
    in_image = cv2.imread(in_filepath, cv2.IMREAD_UNCHANGED)
    
    if in_image is None:
        raise Exception('ERROR: read_input(..) failed for %s' % (in_filepath,))
        
    return in_image
    
def execute(
    file_id_start, 
    file_id_finish):
        
    n_block_rows = ALIGN_RES_TILE_ROWS / BLOCK_N_ROWS
    n_block_cols = ALIGN_RES_TILE_COLS / BLOCK_N_COLS
    
    for file_id in xrange(file_id_start, file_id_finish):
        
        in_image = read_input(file_id)
        
        pad_3D_file_id = None
        in_image_pad_3D = None
        
        block_depth_id = file_id / BLOCK_N_IMAGES
        
        pos_idx = file_id % BLOCK_N_IMAGES
        
        pad_block_depth_id = None
        if ((block_depth_id > BLOCKS_MIN_DEPTH) and
            (pos_idx >= 0) and
            (pos_idx < Z_PAD)):
            pad_block_depth_id = block_depth_id - 1
            assert(block_depth_id >= 0)
        
        if ((block_depth_id < BLOCKS_MAX_DEPTH) and
            (pos_idx <= (BLOCK_N_IMAGES - 1)) and
            (pos_idx > ((BLOCK_N_IMAGES - 1) - Z_PAD))):
            assert(pad_block_depth_id == None)
            pad_block_depth_id = block_depth_id + 1
        
        for block_row_id in xrange(n_block_rows):
            for block_col_id in xrange(n_block_cols):
                block_name = get_block_name(
                    block_depth_id, 
                    block_row_id, 
                    block_col_id)

                print '-- processing %s' % (block_name,)
                
                rows_start = block_row_id * BLOCK_N_ROWS
                cols_start = block_col_id * BLOCK_N_COLS
                rows_finish = rows_start + BLOCK_N_ROWS
                cols_finish = cols_start + BLOCK_N_COLS
                
                rows_start = max(0, rows_start - CNN_PATCH_LEG - X_PAD)
                cols_start = max(0, cols_start - CNN_PATCH_LEG - Y_PAD)
                rows_finish = min(ALIGN_RES_TILE_ROWS, rows_finish + CNN_PATCH_LEG + X_PAD)
                cols_finish = min(ALIGN_RES_TILE_COLS, cols_finish + CNN_PATCH_LEG + Y_PAD)
                
                print ' -- get [%d - %d , %d - %d] image' % (
                    rows_start,
                    rows_finish,
                    cols_start,
                    cols_finish)
                
                block_image = in_image[rows_start:rows_finish,cols_start:cols_finish]                    
                
                block_em_path = get_block_em_path(
                    block_depth_id, 
                    block_row_id, 
                    block_col_id)
                
                if not os.path.exists(block_em_path):
                    print ' -- creating em dir: %s' % (block_em_path,)
                    try:
                        os.makedirs(block_em_path)
                    except OSError, e:
                        if e.errno != 17:
                            raise e
                
                out_filename = '%s_block_id_%.4d_%.4d_%.4d.png' % (
                    PREFIX, 
                    file_id,
                    block_row_id,
                    block_col_id) 
                    
                out_filepath = os.path.join(block_em_path, out_filename)
                print ' -- write: %s' % (out_filepath,)
                cv2.imwrite(out_filepath, block_image)
                
                if pad_block_depth_id is not None:
                    print ' -- processing padding on Z'
                    block_name = get_block_name(
                        pad_block_depth_id, 
                        block_row_id, 
                        block_col_id)
                    
                    block_em_path = get_block_em_path(
                        pad_block_depth_id, 
                        block_row_id, 
                        block_col_id)

                    if not os.path.exists(block_em_path):
                        print ' -- creating em dir: %s' % (block_em_path,)
                        try:
                            os.makedirs(block_em_path)
                        except OSError, e:
                            if e.errno != 17:
                                raise e
                    
                    out_filepath = os.path.join(block_em_path, out_filename)
                    print ' -- write Z pad: %s' % (out_filepath,)
                    cv2.imwrite(out_filepath, block_image)
                
    
if '__main__' == __name__:
    try:
        (prog_name, 
         file_id_start, 
         file_id_finish) = sys.argv[:3]
        
        file_id_start = int(file_id_start)
        file_id_finish = int(file_id_finish)
        
    except ValueError, e:
        sys.exit('USAGE: %s \
    [file_id_start] \
    [file_id_finish]' % (sys.argv[0],))
    
    start_time_secs = time.time()
    
    execute(
        file_id_start, 
        file_id_finish)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ELAPSED TIME: %s seconds' % (elapsed_time_secs,)
    
    print PROC_SUCCESS_STR
    

