
import os
import sys
import time

import cv2
import numpy as np

from pipeline_common import *
  
def execute(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_meta_path = get_block_meta_path(
        block_depth,
        block_row_id,
        block_col_id)
    
    block_probs_processed_path = get_block_probs_processed_path(
        block_depth,
        block_row_id,
        block_col_id)
    
    print ' -- block_meta_path   : %s' % (block_meta_path,)
    print ' -- block_probs_processed_path : %s' % (block_probs_processed_path,)
    
    block_probs_ws_path = get_block_probs_ws_path(    
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_probs_np_path = get_block_probs_np_path(    
        block_depth, 
        block_row_id, 
        block_col_id)
    
    verify_block_out_dir(block_probs_ws_path)
    verify_block_out_dir(block_probs_np_path)
    
    ws_np_prepare_cmd = ('%s %d %s %s %s %s' % 
        (WS_NP_PREPARE_BIN_PATH,
         CNN_PATCH_LEG,
         block_meta_path,
         block_probs_processed_path,
         block_probs_ws_path,
         block_probs_np_path))
        
    (is_success, out_lines) = exec_cmd(ws_np_prepare_cmd)
    
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
