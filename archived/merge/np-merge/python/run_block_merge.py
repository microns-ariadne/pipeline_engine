
import os
import sys
import time

import cv2
import numpy as np

from pipeline_common import *

IS_FORCE = True
IS_SKIP_MERGE_VI = True

Z_WIDTH = 64
X_WIDTH = 256
Y_WIDTH = 256

Z_DIRECTION = 0
X_DIRECTION = 1
Y_DIRECTION = 2

def exec_merge_pair(
    block_depth,
    block_row_id,
    block_col_id,
    direction,
    width):
    
    merge_meta_path = get_merge_meta_path()
    
    is_force_str = ''
    if IS_FORCE:
        is_force_str = '--force'
    
    is_skip_merge_vi_str = ''
    if IS_SKIP_MERGE_VI:
        is_skip_merge_vi_str = '--skip-merge-vi'
    
    print '===================================================================='
    print '  -- Execute merge-pair for block [%d,%d,%d] in direction %d' % (
        block_depth,
        block_row_id,
        block_col_id,
        direction)
    print '  -- params:'
    print '    is_force         : %s' % (IS_FORCE,)
    print '    is_skip_merge_vi : %s' % (IS_SKIP_MERGE_VI,)
    print '    width            : %d' % (width)
    print '===================================================================='
    
    merge_pair_exec = ('%s %s %s \
        --meta-dir %s \
        --width %d \
        --block %d %d %d \
        --dir=%d \
        --np-predict %s \
        --np-args="%s"' % 
        (MERGE_PAIR_BIN_PATH,
         is_force_str,
         is_skip_merge_vi_str,
         merge_meta_path,
         width,
         block_depth,
         block_row_id,
         block_col_id,
         direction,
         MERGE_BLOCK_NP_BINARY,
         MERGE_BLOCK_NP_THRESHOLD_PARAM))
         
    (is_success, out_lines) = exec_cmd(merge_pair_exec)
    
    if not is_success:
        raise Exception('ERROR: exec failed for block [%d,%d,%d] in direction %d' % (
            block_depth,
            block_row_id,
            block_col_id,
            direction))
        
    print '===================================================================='
    print '-- EXEC SUCCEEDED'
    print '===================================================================='
    
    
def execute(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    # Z
    exec_merge_pair(
        block_depth,
        block_row_id,
        block_col_id,
        Z_DIRECTION,
        Z_WIDTH)
    
    # X
    exec_merge_pair(
        block_depth,
        block_row_id,
        block_col_id,
        X_DIRECTION,
        X_WIDTH)
    
    # Y
    exec_merge_pair(
        block_depth,
        block_row_id,
        block_col_id,
        Y_DIRECTION,
        Y_WIDTH)
        
        
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
    
    execute(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ELAPSED TIME: %s seconds' % (elapsed_time_secs,)
    
    print PROC_SUCCESS_STR
