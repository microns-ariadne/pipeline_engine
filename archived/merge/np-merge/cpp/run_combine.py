
import os
import sys
import time

import cv2
import numpy as np

from pipeline_common import *

VI_RATIO = 0.1

def exec_combine():
    
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
    
    
def execute():
    
    print '===================================================================='
    print '  -- Execute combine'
    print '  -- params:'
    print '    VI_ratio         : %.2f' % (VI_RATIO,)
    print '===================================================================='
    
    merge_meta_path = get_merge_meta_path()
    
    combine_exec = ('%s \
        --meta-dir %s \
        --vi-ratio %.2f' % 
        (MERGE_COMBINE_BIN_PATH,
         merge_meta_path,
         VI_RATIO))
         
    (is_success, out_lines) = exec_cmd(combine_exec)
    
    if not is_success:
        raise Exception('ERROR: combine exec failed')
    
    print '===================================================================='
    print '-- EXEC SUCCEEDED'
    print '===================================================================='
    
    print '===================================================================='
    print '  -- Execute combine maxIDs'
    print '===================================================================='
    
    combine_maxIDs_exec = ('%s' % 
        (MERGE_COMBINE_MAXIDS_BIN_PATH))
    
    (is_success, out_lines) = exec_cmd(combine_maxIDs_exec)
    
    if not is_success:
        raise Exception('ERROR: combine maxIDs exec failed')
    
    print '===================================================================='
    print '-- EXEC SUCCEEDED'
    print '===================================================================='
    
        
if '__main__' == __name__:
    try:
        (prog_name,) = sys.argv[:4]
                
    except ValueError, e:
        sys.exit('USAGE: %s' % (sys.argv[0],))
    
    start_time_secs = time.time()
    
    execute()
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ELAPSED TIME: %s seconds' % (elapsed_time_secs,)
    
    print PROC_SUCCESS_STR
