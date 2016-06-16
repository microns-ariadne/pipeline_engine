import argparse
import time
import os
import re
import sys
import shutil

import cv2
import cilk_rw
import numpy as np

from util import *

# Helper classes
class InPaths:
    pass
    
class OutPaths:
    pass

if __name__ == '__main__':
    
    ################################################################# 
    # Setup
    #################################################################
        
    parser = argparse.ArgumentParser('Preprocess the given block and write results in working directory')

    parser.add_argument('--block',
                        nargs=3,
                        dest='block',
                        type=int,
                        required=True,
                        help='Block to preprocess')

    parser.add_argument('--force',
                        dest='is_force',
                        action='store_true',
                        help='Rewrite files.')

    args = parser.parse_args()
    
    block_depth_id = args.block[0]
    block_row_id = args.block[1]
    block_col_id = args.block[2]
    
    InPaths.segmBlockDir = get_block_np_path(
        block_depth_id,
        block_row_id,
        block_col_id)
    
    InPaths.npProbsBlockDir = get_block_probs_np_path(
        block_depth_id,
        block_row_id,
        block_col_id)
    
    InPaths.wsProbsBlockDir = get_block_probs_ws_path(
        block_depth_id,
        block_row_id,
        block_col_id)
    
    args.work_dir = get_block_merge_path(
        block_depth_id, 
        block_row_id, 
        block_col_id)
    
    print 'INPUT PATHS:'
    print '  -- NP       : %s' % (InPaths.segmBlockDir,)
    print '  -- PROBS_NP : %s' % (InPaths.npProbsBlockDir,)
    print '  -- PROBS_WS : %s' % (InPaths.wsProbsBlockDir,)
    
    OutPaths.segmDir    = os.path.join(args.work_dir, 'segmentation_{z:04d}_{y:04d}_{x:04d}'.format(z=args.block[0], y=args.block[1], x=args.block[2]))
    OutPaths.npProbsDir = os.path.join(args.work_dir, 'probs_np_{z:04d}_{y:04d}_{x:04d}'.format(z=args.block[0], y=args.block[1], x=args.block[2]))
    OutPaths.wsProbsDir = os.path.join(args.work_dir, 'probs_ws_{z:04d}_{y:04d}_{x:04d}'.format(z=args.block[0], y=args.block[1], x=args.block[2]))
    
    print 'OUTPUT PATHS:'
    print '  -- NP       : %s' % (OutPaths.segmDir,)
    print '  -- PROBS_NP : %s' % (OutPaths.npProbsDir,)
    print '  -- PROBS_WS : %s' % (OutPaths.wsProbsDir,)
    
    ################################################################# 
    # Clean
    #################################################################

    if args.is_force:
        print 'Force is ENABLED: Deleting files for %s' % (str(args.block))
        
        print '  -- clean: %s' % (OutPaths.segmDir,)
        verify_block_out_dir(OutPaths.segmDir)
             
        print '  -- clean: %s' % (OutPaths.npProbsDir,)
        verify_block_out_dir(OutPaths.npProbsDir)
        
        print '  -- clean: %s' % (OutPaths.wsProbsDir,)
        verify_block_out_dir(OutPaths.wsProbsDir)
        
    ################################################################# 
    # Start
    #################################################################
    
    print 'STAR preprocessing block %s' % (str(args.block))
    startTime = time.time()
    
    ################################################################# 
    # Process padding
    #################################################################
    
    D, R, C = args.block[0], args.block[1], args.block[2]

    files = os.listdir(InPaths.segmBlockDir)
    lastD = len(files)
    im = cv2.imread(os.path.join(InPaths.segmBlockDir, files[0]), cv2.IMREAD_UNCHANGED)
    lastR = im.shape[0]
    lastC = im.shape[1]

    firstD = firstR = firstC = 0 
    
    z_pad = Z_PAD - 1 - MERGE_PREPROCESS_Z_OVERLAP
    x_pad = X_PAD / BLOCK_SUB_SAMPLE
    y_pad = Y_PAD / BLOCK_SUB_SAMPLE
    if D > BLOCKS_MIN_DEPTH:
        firstD += z_pad
    if D < BLOCKS_MAX_DEPTH:
        lastD -= z_pad

    if R > BLOCKS_MIN_ROW_ID:
        firstR += x_pad
    if R < BLOCKS_MAX_ROW_ID:
        lastR -= x_pad

    if C > BLOCKS_MIN_COL_ID:
        firstC += y_pad
    if C < BLOCKS_MAX_COL_ID:
        lastC -= y_pad
    
    print ' ----------------------------------------------------------'
    print ' -- PADDING CUTS [D/R/C]: [%d-%d] [%d-%d] [%d-%d]' % (
        firstD, lastD,
        firstR, lastR,
        firstC, lastC)
        
    ################################################################# 
    # Process NP
    #################################################################
    
    print 'Input np-dir: %s' % (InPaths.segmBlockDir,)
    
    segmentation = cilk_rw.read_rgb_labels(InPaths.segmBlockDir)

    segmentation = segmentation[firstD : lastD, firstR : lastR, firstC : lastC]
    
    # Compress segmentation
    index_mapping = np.cumsum(np.bincount(segmentation.ravel()) > 0, dtype= np.uint32)
    index_mapping[0] = 0
    maxId = index_mapping.max()
    segmentation = index_mapping[segmentation]

    cilk_rw.write_labels_rgb(
        segmentation, 
        OutPaths.segmDir, 
        'segm_%04d_%04d_%04d-compressed' % (args.block[0], args.block[1], args.block[2]))
    
    with open(os.path.join(args.work_dir, 'segmentation_%04d_%04d_%04d_maxID.txt') % (args.block[0], args.block[1], args.block[2]), 'wb') as f:
        f.write(str(maxId))

    ################################################################# 
    # Process PROBS_NP
    #################################################################
    
    print 'Input probs-np-dir: %s' % (InPaths.npProbsBlockDir,)
    
    probs = cilk_rw.read_probabilities_int(InPaths.npProbsBlockDir)
    
    probs = probs[firstD : lastD, firstR : lastR, firstC : lastC]
    
    cilk_rw.write_int_probabilities(
        probs, 
        OutPaths.npProbsDir, 
        'probs_np_%04d_%04d_%04d' % (D, R, C))

    ################################################################# 
    # Process PROBS_WS
    #################################################################
    
    print 'Input probs-ws-dir: %s' % (InPaths.wsProbsBlockDir,)
    
    probs = cilk_rw.read_probabilities_int(InPaths.wsProbsBlockDir)
    
    probs = probs[firstD : lastD, firstR : lastR, firstC : lastC]
    
    cilk_rw.write_int_probabilities(
        probs, 
        OutPaths.wsProbsDir, 
        'probs_ws_%04d_%04d_%04d' % (D, R, C))

    ################################################################# 
    # Done
    #################################################################
    
    print 'DONE: Preprocessing %s' % (InPaths.segmBlockDir,)
    print '    Time: %f' % (time.time() - startTime)
    
    print PROC_SUCCESS_STR
    

