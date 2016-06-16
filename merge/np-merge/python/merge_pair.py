import argparse
import time
import os
import sys

import cilk_rw
import numpy as np

from util import *
import slicing
import merging

def process_slice_supv(args):
    
    print 'process_slice_supv: start'
    
    if os.path.exists(args.sliceSupvPath) and len(os.listdir(args.sliceSupvPath)) > 0:
        print ' -- Already exists: read slice supervoxels from %s' % (args.sliceSupvPath,)
        
        slice_supv = cilk_rw.read_rgb_labels(args.sliceSupvPath)
        
        with open(args.sliceSupvPath + '_shift.txt', 'rb') as f:
            shift = int(f.readline())
            
        print 'process_slice_supv: finish'
        
        return (slice_supv, shift)
    
    print ' -- Generating slice supervoxels'
    
    print '   -- Read supervoxels'
    segmentations = [ ]
    for i, path in enumerate(args.segmPaths):
        print '   -- [%d] %s' % (i, path,)
        segmentations.append(cilk_rw.read_rgb_labels(path))
    
    print '   -- Extract slice supervoxels'

    # sliceDirection = 2 - args.direction 
    sliceDirection = args.direction

    slice_supv = slicing.slice_data(
        segmentations[0], 
        segmentations[1], 
        args.width, 
        sliceDirection)

    slice_coord = [slice(None)] * len(slice_supv.shape)
    slice_coord[sliceDirection] = slice(None, args.width)

    shift = slice_supv[slice_coord].max() + np.array(1, dtype = np.uint32)

    slice_coord[sliceDirection] = slice(args.width, None)

    slice_supv[slice_coord] += shift * (slice_supv[slice_coord] > 0)
    
    print '   -- Write slice supervoxels'
    
    cilk_rw.write_labels_rgb(
        slice_supv, 
        args.sliceSupvPath, 
        os.path.basename(args.sliceSupvPath))
    
    print '   -- Write shift of slice supervoxels'    
    with open(args.sliceSupvPath + '_shift.txt', 'w') as f:
        f.write(str(shift))
    
    print 'process_slice_supv: finish'
    
    return (slice_supv, shift)

def process_slice_probs(args):
    
    print 'process_slice_probs: start'
    
    if os.path.exists(args.sliceProbsPath) and len(os.listdir(args.sliceProbsPath)) > 0:
        print ' -- SKIP: slice probs exists %s' % (args.sliceSegmPath,)
        print 'process_slice_probs: finish'
        return
    
    print ' -- Generating slice probs'
    
    print '   -- Read probabilities'
    probs = [ ]
    for i, path in enumerate(args.npProbsPaths):
        print '   -- [%d] %s' % (i, path,)
        probs.append(cilk_rw.read_probabilities_int(path))
    
    print '   -- Extract slice probability'
    
    probs_data = slicing.slice_data(
        probs[0], 
        probs[1], 
        args.width, 
        args.direction)
    
    print '   -- Write slice probability'
        
    cilk_rw.write_int_probabilities(
        probs_data, 
        args.sliceProbsPath, 
        os.path.basename(args.sliceProbsPath))
    
    print 'process_slice_probs: finish'
    
def process_slice_segm(args):
    
    print 'process_slice_segm: start'
    
    if os.path.exists(args.sliceSegmPath) and len(os.listdir(args.sliceSegmPath)) > 0:
        print ' -- SKIP: slice segmentation exists %s' % (args.sliceSegmPath,)
        
        print 'process_slice_segm: finish'
        
        slice_segm = cilk_rw.read_rgb_labels(args.sliceSegmPath)
        
        return slice_segm
    
    assert os.path.exists(args.sliceProbsPath), 'Slice probabilities not found: %s' % (args.sliceProbsPath,)
    assert os.path.exists(args.sliceSupvPath), 'Slice supervoxels not found: %s' % (args.sliceSupvPath,)
    
    print ' -- Generating slice segmentation'
    
    print '   -- Execute NP'
    
    np_cmd = '%s %s %s %s --output-file %s/slice_segm_ %s' % (
        args.np_predict, 
        args.sliceSupvPath, 
        args.sliceProbsPath, 
        args.classifier, 
        args.sliceSegmPath, 
        args.np_args)
    
    (is_success, out_lines) = exec_cmd(np_cmd)
    
    if not is_success:
        raise Exception('Failed NP_CMD: %s' % (np_cmd,))
    
    slice_segm = cilk_rw.read_rgb_labels(args.sliceSegmPath)
    
    print 'process_slice_segm: finish'
    
    return slice_segm
    
    
if __name__ == '__main__':
    
    ################################################################# 
    # Setup
    #################################################################
    
    parser = argparse.ArgumentParser('Given a block index triple and an axis, merges the given block with its neighbour in the direction of the axis.')
    
    parser.add_argument('--meta-dir',
            dest='meta_dir',
            type=str,
            required=True,
            help='Directory to keep meta results in.')

    parser.add_argument('--block',
            nargs=3,
            dest='block',
            type=int,
            required=True,
            help='The index of the block to be merged with its neighbour.')

    parser.add_argument('--dir',
            dest='direction',
            type=int,
            required=True,
            help='The direction on which to merge. One int, 0, 1 or 2 indicating which neighbour of block should be used. If block is (z, y, x) and direction = 1, \
                  then blocks (z, y, x) and (z, y + 1, x) will be merged')

    parser.add_argument('--width',
            dest='width',
            type=int,
            required=True,
            help='The width of slice.')

    parser.add_argument('--classifier',
            dest='classifier',
            type=str,
            help='The classifier for merging. Each default classifier is used according to the direction of merge.')

    parser.add_argument('--threshold',
            dest='threshold',
            type=int,
            help='Consider pairs which have common boundary of size at least threshold for VI merge. Each direction has a default value.')

    parser.add_argument('--vi_classifier',
            dest='vi_classifier',
            type=str,
            default=CLASSIFIER,
            help='The classifier to use to run segmentation on slice (to be used for VI mergin). Can be the general classifier used to segment blocks.')

    parser.add_argument('--force',
            dest='is_force',
            action='store_true',
            help='Rewrite files.')

    parser.add_argument('--np-predict',
            dest='np_predict',
            type=str,
#            default=NP_PREDICT,
            help='Path to NeuroProof.')

    parser.add_argument('--np-args',
            dest='np_args',
            type=str,
            default='',
            help='Extra arguments to pass to neuroproof')

    parser.add_argument('--skip-merge-vi',
            dest='skip_merge_vi',
            action='store_true',
            help='If set, will not merge using VI values')
    
    args = parser.parse_args()

    if args.direction < 0 or args.direction > 2:
        print 'ERROR: Direction has to be 0, 1 or 2.'
        parser.print_help()
        sys.exit(0)
    
    args.np_predict = os.path.abspath(args.np_predict)

    if args.classifier == None:
        args.classifier = SLICE_CLASSIFIERS[args.direction]
    if args.threshold == None:
        args.threshold = HEURISTIC_THRESHOLD[args.direction]

    args.classifier = os.path.abspath(args.classifier)
    args.vi_classifier = os.path.abspath(args.vi_classifier)

    if not os.path.isdir(args.meta_dir):
        raise Exception('Invalid meta-dir: %s' % (args.meta_dir,))
    
    otherBlock = list(args.block)
    otherBlock[args.direction] += 1

    args.blocks = [tuple(args.block), tuple(otherBlock)]
    
    block_dir_1 = get_block_merge_path(
        args.block[0], 
        args.block[1], 
        args.block[2])
    
    block_dir_2 = get_block_merge_path(
        otherBlock[0], 
        otherBlock[1], 
        otherBlock[2])
    
    block_dirs = [block_dir_1, block_dir_2]
        
    for block_dir in block_dirs:
        if not os.path.isdir(block_dir):
            raise Exception('Invalid block dir: %s' % (block_dir,))
    
    args.segmPaths    = [chunk_dir_path(block_dirs[i], block, 'segmentation') 
                         for i, block in enumerate(args.blocks)]
    
    args.npProbsPaths = [ ]
    args.wsProbsPaths = [ ]
    for i, block in enumerate(args.blocks):
        block_dir = block_dirs[i]
        
        args.npProbsPaths.append(
            os.path.join(
                block_dir,
                'probs_np_%04d_%04d_%04d' % (block[0], block[1], block[2])))
        args.wsProbsPaths.append(
            os.path.join(
                block_dir,
                'probs_ws_%04d_%04d_%04d' % (block[0], block[1], block[2])))
    
    if None in args.segmPaths:
        print 'NO_MERGE: Input block is %s, direction is %d' % (args.blocks[0], args.direction)
        print '   -- Block %s does not exist' % str(args.blocks[args.segmPaths.index(None)])
        print '   -- EXITING'
        print PROC_SUCCESS_STR
        sys.exit(0)

    args.sliceProbsPath = slice_dir(args, block_dir_1, 'probs_np')
    args.sliceSupvPath  = slice_dir(args, block_dir_1, 'supv') # supervoxles paths from segmentations of blocks
    args.sliceSegmPath  = slice_dir(args, block_dir_1, 'segmentation') # segmentation of slice using supervoxels sliceSupvPath

    args.sliceWsProbsPath = slice_dir(args, block_dir_1, 'probs_ws') # probs for slice for watershed
    args.sliceWsPath      = slice_dir(args, block_dir_1, 'watershed') # path of watershed gotten from prob map
    args.sliceNPPath      = slice_dir(args, block_dir_1, 'np') # segmentation of slice using supervoxels from sliceWsPath

    args.mergePath = merge_path(args)
    
    ################################################################# 
    # Clean
    #################################################################
    
    if args.is_force:
        print 'Force mode is enabled: Deleting files for block %s and direction %d' % (
            str(args.blocks[0]), 
            args.direction)
            
        verify_block_out_dir(args.sliceProbsPath)
        verify_block_out_dir(args.sliceSupvPath)
        verify_block_out_dir(args.sliceSegmPath)
        verify_block_out_dir(args.sliceWsProbsPath)
        verify_block_out_dir(args.sliceWsPath)
        verify_block_out_dir(args.sliceNPPath)
        if os.path.exists(args.mergePath):
            os.unlink(args.mergePath)
    
    if os.path.isfile(args.mergePath):
        print 'SKIPPING: Merge File exists %s' % args.mergePath
        print '    Please remove to calculate, or use --force to recalculate everything.'
        sys.exit(0)
    
    ################################################################# 
    # Start Basic
    #################################################################
    
    timeStart = time.time()
    
    print 'START: Merging block %s, direction %d' % (
        str(args.blocks[0]), 
        args.direction)
    
    process_slice_probs(args)
    
    (slice_supv, shift) = process_slice_supv(args)
    
    slice_segm = process_slice_segm(args)
    
    print ' -- Generate output file: %s' % (args.mergePath,)
    out_file = open(args.mergePath, 'wb')
    
    print ' -- Start merge pairs'

    # merge_pairs = merging.get_merge_pairs(slice_supv, slice_segm, shift, args.width, 2 - args.direction).transpose()
    merge_pairs = merging.get_merge_pairs(
        slice_supv, 
        slice_segm, 
        shift, 
        args.width, 
        args.direction)
    
    merge_pairs = merge_pairs.transpose()
    
    out_file.write(str(merge_pairs.shape[0]) + '\n')
    np.savetxt(out_file, merge_pairs, fmt='%u')
    
    print ' -- Done merge pairs'
    
    ################################################################# 
    # Start VI
    #################################################################
    
    if not args.skip_merge_vi:
        print 'Using VI to filter heuristic pairs'

        vi_start = time.time()
        #if h5py.is_hdf5(args.sliceNPPath):
        if os.path.exists(args.sliceNPPath):
            print 'Slice NP segmentation from probabilities directly exists: %s' % (args.sliceNPPath)
        else:
            print ' -- Generate slice NP segmentation from probabilities'
            # if h5py.is_hdf5(args.sliceWsPath):
            if os.path.exists(args.sliceWsPath):
                print 'Slice WS from raw probability maps exists: %s' % args.sliceWsPath
            else:
                print ' -- Generate slice WS from probabilities'
                # if h5py.is_hdf5(args.sliceWsProbsPath):
                if os.path.exists(args.sliceWsProbsPath):
                    print 'Slice WS probs exists: %s' % (args.sliceWsProbsPath)
                else:
                    print ' -- Generating slice from ws_probability maps'

                    wsProbs = [ ]
                    for path in args.wsProbsPaths:
                        print ' -- Read WS probs: %s' % (path,)
                        # with h5py.File(path) as f:
                        #     wsProbs.append(np.array(f['volume/predictions'], dtype = np.float32))
                        wsProbs.append(cilk_rw.read_probabilities_int(path))
                    
                    print ' -- Slice WS probs'
                    wsProbsSlice = slicing.slice_data(wsProbs[0], wsProbs[1], args.width, args.direction)
                    cilk_rw.write_int_probabilities(wsProbsSlice, args.sliceWsProbsPath, os.path.basename(args.sliceWsProbsPath))
                    # with h5py.File(args.sliceWsProbsPath, 'w') as f:
                    #     f.create_dataset('volume/predictions', data=wsProbsSlice, compression = H5_COMPRESSION)
                
                print ' -- Execute WS'
                exec_cmd('mkdir %s' % args.sliceWsPath)
                exec_cmd('%s %s %s' % (WATERSHED, args.sliceWsProbsPath, args.sliceWsPath))
            
            if not os.path.exists(args.sliceWsPath):
                print 'ERROR: watershed failed (no file %s)' % args.sliceWsPath
                sys.exit(0)
                
            print ' -- Running NP on slice from watershed on probability maps'
            exec_cmd('mkdir %s' % args.sliceNPPath)
            exec_cmd('%s %s %s %s --output-file %s/slice_np_ %s' % (args.np_predict, args.sliceWsPath, args.sliceProbsPath, args.vi_classifier, args.sliceNPPath, args.np_args))
            
            if not os.path.exists(args.sliceNPPath):
                print 'ERROR: NP failed (no file %s)' % args.sliceNPPath
                sys.exit(0)
            
            # print ' -- Compress NP H5 result'
            # slicing.compress_h5(args.sliceNPPath, 'stack')
        
        print ' -- Read NP H5 result: %s' % (args.sliceNPPath,)
        # with h5py.File(args.sliceNPPath) as f:
        #     slice_np = np.array(f['stack'], dtype='uint32')
        slice_np = cilk_rw.read_rgb_labels(args.sliceNPPath)

        print ' -- Start heuristic_merge_pairs'
        heuristic_pairs = merging.get_heuristic_merge_pairs(
                    slice_supv,
                    slice_segm,
                    shift,
                    args.width,
                    #2 - args.direction,
                    args.direction,
                    args.threshold).transpose()

        delta_vi = merging.get_delta_vi_for_pairs(slice_segm, slice_np, heuristic_pairs[..., :2], shift)

        out_file.write('\n' + str(heuristic_pairs.shape[0]) + '\n')
        for i in xrange(len(heuristic_pairs)):
            out_file.write('%u %u %u %f %f\n' % (heuristic_pairs[i][0],
                                                    heuristic_pairs[i][1],
                                                    heuristic_pairs[i][2],
                                                    delta_vi[i][0],
                                                    delta_vi[i][1]))

        out_file.write('\n')
        print 'Done: VI heuristic pairs; Time: %f' % (time.time() - vi_start)

    else:
        print 'Skipped VI merge'
        out_file.write('0\n')
    
    ################################################################# 
    # Done
    #################################################################
    
    out_file.close()
    print
    print 'DONE: Merging block %s direction %d' % (str(args.blocks[0]), args.direction)
    print '    Time: %f' % (time.time() - timeStart)
    print
    print PROC_SUCCESS_STR
 
