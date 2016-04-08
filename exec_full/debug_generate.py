
import sys
import os
import time
import shutil

import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import h5py

from scipy.ndimage.morphology import grey_dilation

IS_GENERATE_DATA = 0

IS_GENERATE_PROBS = 0

IS_GENERATE_PROBS_WS = 0

IS_GENERATE_PROBS_NP = 0

IS_GENERATE_WS = 0

IS_GENERATE_NP = 0

IS_GENERATE_NP_MERGE = 1

def get_segments(block_path):
    filenames = os.listdir(block_path)
    
    seg_names = []

    for filename in filenames:
        if not os.path.isdir(os.path.join(block_path, filename)):
            continue
        if filename.find('_segment_') == -1:
            continue
        if filename.find('_type_complex') != -1:
            continue

        seg_names.append(filename)

    seg_names.sort()

    res_segments = []
    for seg_name in seg_names:

        parts = seg_name.split('_range_')[1].split('_type_')[0].split('_')
        start_id = int(parts[0])
        finish_id = int(parts[1]) + 1
        
        res_segments.append((seg_name, start_id, finish_id))

    return res_segments
    
def is_segments_match(segments_1, segments_2):
    
    assert(len(segments_1) == len(segments_2))
    
    for i in xrange(len(segments_1)):
        segment_1 = segments_1[i]
        segment_2 = segments_2[i]
        
        seg_name_1 = segment_1[0].split('_type_')[0]
        start_id_1 = segment_1[1]
        finish_id_1 = segment_1[2]
        
        seg_name_2 = segment_2[0].split('_type_')[0]
        start_id_2 = segment_2[1]
        finish_id_2 = segment_2[2]
        
        assert(start_id_1 == start_id_2)
        assert(finish_id_1 == finish_id_2)
        assert(seg_name_1 == seg_name_2)
        
    return True
    
def cnn_fix_for_block_sizes(cnn_patch_leg, 
                            block_n_depth, 
                            block_n_rows, 
                            block_n_cols):
    
    block_n_rows += cnn_patch_leg * 2
    block_n_cols += cnn_patch_leg * 2
    
    return (block_n_depth, block_n_rows, block_n_cols)

def debug_generate_data(
    block_path_debug,
    out_prefix,
    block_Z,
    block_X,
    block_Y,
    block_path_data, 
    segments,
    block_n_depth,
    block_n_rows,
    block_n_cols,
    cnn_patch_leg):

    print 'debug_generate_data: start'
    
    (cnn_block_n_depth,
     cnn_block_n_rows,
     cnn_block_n_cols) = cnn_fix_for_block_sizes(cnn_patch_leg, block_n_depth, block_n_rows, block_n_cols)
    
    print '  -- block_size [ORIG_SIZE]: [%d,%d,%d]' % (block_n_depth, block_n_rows, block_n_cols)
    print '  -- block_size [CNN_FIXED]: [%d,%d,%d]' % (cnn_block_n_depth, cnn_block_n_rows, cnn_block_n_cols)
    
    block_data = np.zeros((block_n_depth, block_n_rows, block_n_cols), dtype = np.uint8)
    
    for segment in segments:
        seg_name = segment[0]
        start_id = segment[1]
        finish_id = segment[2]
        
        print '  -- segment[%d-%d]: %s' % (start_id, finish_id, seg_name,)
        
        seg_path_data = os.path.join(block_path_data, '%s' % (seg_name,))
        
        seg_files = os.listdir(seg_path_data)
        seg_files.sort()
        
        for file_idx, seg_file in enumerate(seg_files):
            seg_filepath = os.path.join(seg_path_data, seg_file)
            
            print '  -- Read[%d]: %s' % (file_idx, seg_filepath)
            
            im_data = cv2.imread(seg_filepath, cv2.IMREAD_UNCHANGED)
            
            block_data[start_id + file_idx, :, :] = im_data[cnn_patch_leg:-cnn_patch_leg, cnn_patch_leg:-cnn_patch_leg]
            
    block_path_debug_data = os.path.join(block_path_debug, 'data')
    
    if not os.path.exists(block_path_debug_data):
        os.makedirs(block_path_debug_data)
        
    for cur_depth in xrange(block_n_depth):
        im_data = block_data[cur_depth, :, :]
        
        out_filename = '%s_block_%.4d_%.4d_%.4d_%.4d_data.png' % (
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            cur_depth,)
            
        out_filepath = os.path.join(block_path_debug_data, out_filename)
        
        print '  -- Write[%d]: %s' % (cur_depth, out_filepath)
        
        cv2.imwrite(out_filepath, im_data)
    
    
    print 'debug_generate_data: finish'

def debug_generate_probs(
    block_path_debug,
    out_dirname,
    out_prefix,
    block_Z,
    block_X,
    block_Y,
    block_path_probs, 
    segments_probs,
    block_n_depth,
    block_n_rows,
    block_n_cols):

    print 'debug_generate_probs: start [out_dirname = %s]' % (out_dirname,)

    print '  -- block_size: [%d,%d,%d]' % (block_n_depth, block_n_rows, block_n_cols)

    block_data = np.zeros((block_n_depth, block_n_rows, block_n_cols), dtype = np.uint8)

    for segment in segments_probs:
        seg_name = segment[0]
        start_id = segment[1]
        finish_id = segment[2]

        print '  -- segment[%d-%d]: %s' % (start_id, finish_id, seg_name,)

        seg_path_probs = os.path.join(block_path_probs, '%s' % (seg_name,))

        seg_files = os.listdir(seg_path_probs)
        seg_files.sort()

        for file_idx, seg_file in enumerate(seg_files):
            seg_filepath = os.path.join(seg_path_probs, seg_file)

            print '  -- Read[%d]: %s' % (file_idx, seg_filepath)

            im_probs = cv2.imread(seg_filepath, cv2.IMREAD_UNCHANGED)

            block_data[start_id + file_idx, :, :] = im_probs

    block_path_debug_probs = os.path.join(block_path_debug, out_dirname)

    if not os.path.exists(block_path_debug_probs):
        os.makedirs(block_path_debug_probs)

    for cur_depth in xrange(block_n_depth):
        im = block_data[cur_depth, :, :]
        
        out_filename = '%s_block_%.4d_%.4d_%.4d_%.4d_%s.png' % (
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            cur_depth,
            out_dirname,)
        
        out_filepath = os.path.join(block_path_debug_probs, out_filename)
        
        print '  -- Write[%d]: %s' % (cur_depth, out_filepath)
        
        cv2.imwrite(out_filepath, im)
        
    print 'debug_generate_probs: finish [out_dirname = %s]' % (out_dirname,)

# def normalize_3D_block_uint32(block_data, in_min = None, in_max = None):
#     block_data = block_data.astype(np.float)
#     
#     if in_min == None:
#         min_value = block_data.min()
#     else:
#         min_value = in_min
#     
#     block_data -= min_value
#     
#     if in_max == None:
#         max_value = block_data.max()
#     else:
#         max_value = in_max
#     
#     block_data /= max_value
#     
#     block_data *= (np.iinfo(np.uint32).max - 10000)
#     
#     block_data = block_data.astype(np.uint32) + 100
#     
#     return block_data
 
def normalize_3D_block_uint32(block_data):
    block_data = block_data.astype(np.uint64)
    block_data *= 1000000 # Magic
    block_data %= 16777213 # the largest prime that is smaller than a cube of 256
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

def debug_generate_ws(
    block_path_debug,
    out_prefix,
    block_Z,
    block_X,
    block_Y,
    block_path_ws, 
    segments_ws,
    block_n_depth,
    block_n_rows,
    block_n_cols):

    print 'debug_generate_ws: start'

    print '  -- block_size: [%d,%d,%d]' % (block_n_depth, block_n_rows, block_n_cols)

    block_data = np.zeros((block_n_depth, block_n_rows, block_n_cols), dtype = np.uint32)

    for segment in segments_ws:
        seg_name = segment[0]
        start_id = segment[1]
        finish_id = segment[2]
        
        print '  -- segment[%d-%d]: %s' % (start_id, finish_id, seg_name,)
        
        seg_path_ws = os.path.join(block_path_ws, '%s' % (seg_name,))
        
        seg_files = os.listdir(seg_path_ws)
        seg_files.sort()
        
        for file_idx, seg_file in enumerate(seg_files):
            seg_filepath = os.path.join(seg_path_ws, seg_file)
            
            print '  -- Read[%d]: %s' % (file_idx, seg_filepath)

            im_ws = cv2.imread(seg_filepath, cv2.IMREAD_UNCHANGED)
            
            im_ws[:,:,2] *= (1<<16)
            im_ws[:,:,1] *= (1<<8)
            im_ws[:,:,0] = im_ws[:,:,0] + im_ws[:,:,1] + im_ws[:,:,2]
            
            block_data[start_id + file_idx, :, :] = im_ws[:,:,0] + im_ws[:,:,1] + im_ws[:,:,2]
        
    block_data = normalize_3D_block_uint32(block_data)
        
    block_path_debug_ws = os.path.join(block_path_debug, 'ws')
    
    if not os.path.exists(block_path_debug_ws):
        os.makedirs(block_path_debug_ws)
    
    for cur_depth in xrange(block_n_depth):
        im_ws = block_data[cur_depth, :, :]
            
        im_ws_rgb = get_RGB(im_ws)
            
        out_filename = '%s_block_%.4d_%.4d_%.4d_%.4d_ws.png' % (
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            cur_depth)
        
        out_filepath = os.path.join(block_path_debug_ws, out_filename)
        
        print '  -- Write[%d]: %s' % (cur_depth, out_filepath)
            
        cv2.imwrite(out_filepath, im_ws_rgb)
        
    print 'debug_generate_ws: finish'

def debug_generate_np(
    block_path_debug,
    out_prefix,
    block_Z,
    block_X,
    block_Y,
    block_path_np, 
    segments_np,
    block_n_depth,
    block_n_rows,
    block_n_cols):
    
    print 'debug_generate_np: start'
    
    print '  -- block_size: [%d,%d,%d]' % (block_n_depth, block_n_rows, block_n_cols)
    
    block_data = np.zeros((block_n_depth, block_n_rows, block_n_cols), dtype = np.uint32)
    
    for segment in segments_np:
        seg_name = segment[0]
        start_id = segment[1]
        finish_id = segment[2]
        
        print '  -- segment[%d-%d]: %s' % (start_id, finish_id, seg_name,)
        
        h5_np_seg_path = os.path.join(block_path_np, 
                                      '%s' % (seg_name,), 
                                      '%s_segmentation.h5' % (seg_name))
        
        f = h5py.File(h5_np_seg_path)
        
        seg_data = np.array(f['stack'])
        seg_data = seg_data.transpose((2,1,0))
        block_data[start_id : finish_id, :, :] = seg_data
        
    block_data = normalize_3D_block_uint32(block_data)
    
    block_path_debug_np = os.path.join(block_path_debug, 'np')
    
    if not os.path.exists(block_path_debug_np):
        os.makedirs(block_path_debug_np)
    
    for cur_depth in xrange(block_n_depth):
        im_np = block_data[cur_depth, :, :]
        
        im_np_rgb = get_RGB(im_np)
        
        out_filename = '%s_block_%.4d_%.4d_%.4d_%.4d_np.png' % (
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            cur_depth)
        
        out_filepath = os.path.join(block_path_debug_np, out_filename)
        
        print '  -- Write[%d]: %s' % (cur_depth, out_filepath)
        
        cv2.imwrite(out_filepath, im_np_rgb)

    print 'debug_generate_np: finish'

def np_merge_get_labels_max_min(blocks_dir_np_merge):
    
    labels_map_filepath = os.path.join(blocks_dir_np_merge, 'labels_map.txt')
    f = open(labels_map_filepath, 'rb')
    data_lines = f.readlines()
    f.close()
    
    max_label = None
    min_label = None
    for data_line in data_lines[1:]:
        cur_label = int(data_line.split()[-1])
        
        if max_label == None or max_label < cur_label:
            max_label = cur_label
        
        if min_label == None or min_label > cur_label:
            min_label = cur_label
            
            
    return (min_label, max_label)
    
def debug_generate_np_merge(
        block_path_debug,
        out_prefix,
        block_Z,
        block_X,
        block_Y,
        blocks_dir_np_merge,
        block_n_depth,
        block_n_rows,
        block_n_cols):

    print 'debug_generate_np_merge: start'
    
    #(min_label, max_label) = np_merge_get_labels_max_min(blocks_dir_np_merge)
    
    #print '  -- [min_label, max_label] : [%d] [%d]' % (min_label, max_label,)
    print '  -- block_size: [%d,%d,%d]' % (block_n_depth, block_n_rows, block_n_cols)
    
    block_data = np.zeros((block_n_depth, block_n_rows, block_n_cols), dtype = np.uint32)
    
    h5_seg_np_merge = os.path.join(blocks_dir_np_merge, 'out_segmentation_%.4d_%.4d_%.4d.h5' % (
        block_Z,
        block_X,
        block_Y))
    
    f = h5py.File(h5_seg_np_merge)

    seg_data = np.array(f['stack'])
    seg_data = seg_data.transpose((2,1,0))
    block_data[:,:,:] = seg_data
    
    #block_data = normalize_3D_block_uint32(block_data, in_min = min_label, in_max = max_label)
    block_data = normalize_3D_block_uint32(block_data)
    
    block_path_debug_np_merge = os.path.join(block_path_debug, 'np_merge')

    if not os.path.exists(block_path_debug_np_merge):
        os.makedirs(block_path_debug_np_merge)

    for cur_depth in xrange(block_n_depth):
        im_np_merge = block_data[cur_depth, :, :]
        
        im_np_merge_rgb = get_RGB(im_np_merge)

        out_filename = '%s_block_%.4d_%.4d_%.4d_%.4d_np_merge.png' % (
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            cur_depth)

        out_filepath = os.path.join(block_path_debug_np_merge, out_filename)

        print '  -- Write[%d]: %s' % (cur_depth, out_filepath)

        cv2.imwrite(out_filepath, im_np_merge_rgb)

    print 'debug_generate_np_merge: finish'

def execute(blocks_dir_data,
            blocks_dir_probs,
            blocks_dir_probs_ws,
            blocks_dir_probs_np,
            blocks_dir_ws,
            blocks_dir_np,
            blocks_dir_np_merge,
            blocks_dir_debug,
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            block_n_depth,
            block_n_rows,
            block_n_cols,
            cnn_patch_leg):
    
    block_name = 'block_%.4d_%.4d_%.4d' % (block_Z, block_X, block_Y)
    
    print 'Parameters:'
    print ' -- BLOCK: %s [%d,%d,%d]' % (block_name, block_n_depth, block_n_rows, block_n_cols)
    
    block_path_data = os.path.join(blocks_dir_data, block_name)
    print ' -- DATA: %s' % (block_path_data,)
    assert(os.path.exists(block_path_data))
    segments_data = get_segments(block_path_data)
    
    block_path_probs = os.path.join(blocks_dir_probs, '%s_probs' % (block_name,))
    print ' -- PROBS: %s' % (block_path_probs,)
    assert(os.path.exists(block_path_probs))
    segments_probs = get_segments(block_path_probs)
    assert(is_segments_match(segments_probs, segments_data))
    
    block_path_probs_ws = os.path.join(blocks_dir_probs_ws, '%s_probs' % (block_name,))
    print ' -- PROBS_WS: %s' % (block_path_probs_ws,)
    assert(os.path.exists(block_path_probs_ws))
    segments_probs_ws = get_segments(block_path_probs_ws)
    assert(is_segments_match(segments_probs_ws, segments_data))
    
    block_path_probs_np = os.path.join(blocks_dir_probs_np, '%s_probs' % (block_name,))
    print ' -- PROBS_NP: %s' % (block_path_probs_np,)
    assert(os.path.exists(block_path_probs_np))
    segments_probs_np = get_segments(block_path_probs_np)
    assert(is_segments_match(segments_probs_np, segments_data))
    
    block_path_ws = os.path.join(blocks_dir_ws, '%s_ws' % (block_name,))
    print ' -- WS: %s' % (block_path_ws,)
    assert(os.path.exists(block_path_ws))
    segments_ws = get_segments(block_path_ws)
    assert(is_segments_match(segments_ws, segments_data))
    
    block_path_np = os.path.join(blocks_dir_np, '%s_np' % (block_name,))
    print ' -- NP: %s' % (block_path_np,)
    assert(os.path.exists(block_path_np))
    segments_np = get_segments(block_path_np)
    assert(is_segments_match(segments_np, segments_data))
    
    print ' -- NP_MERGE: %s' % (blocks_dir_np_merge,)        
    assert(os.path.exists(blocks_dir_np_merge))
    
    block_path_debug = os.path.join(blocks_dir_debug, '%s_debug' % (block_name,))        
    print ' -- DEBUG: %s' % (block_path_debug,)
    
    if not os.path.exists(block_path_debug):
        os.makedirs(block_path_debug)
    #else:
    #    shutil.rmtree(block_path_debug)
    
    if IS_GENERATE_DATA:
        debug_generate_data(
            block_path_debug,
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            block_path_data, 
            segments_data,
            block_n_depth,
            block_n_rows,
            block_n_cols,
            cnn_patch_leg)
    
    if IS_GENERATE_PROBS:
        debug_generate_probs(
            block_path_debug,
            'probs',
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            block_path_probs, 
            segments_probs,
            block_n_depth,
            block_n_rows,
            block_n_cols)

    if IS_GENERATE_PROBS_WS:
        debug_generate_probs(
            block_path_debug,
            'probs_ws',
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            block_path_probs_ws, 
            segments_probs_ws,
            block_n_depth,
            block_n_rows,
            block_n_cols)
    
    if IS_GENERATE_PROBS_NP:
        debug_generate_probs(
            block_path_debug,
            'probs_np',
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            block_path_probs_np, 
            segments_probs_np,
            block_n_depth,
            block_n_rows,
            block_n_cols)
    
    if IS_GENERATE_WS:        
        debug_generate_ws(
            block_path_debug,
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            block_path_ws, 
            segments_ws,
            block_n_depth,
            block_n_rows,
            block_n_cols)

    if IS_GENERATE_NP:
        debug_generate_np(
            block_path_debug,
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            block_path_np, 
            segments_np,
            block_n_depth,
            block_n_rows,
            block_n_cols)

    if IS_GENERATE_NP_MERGE:
        debug_generate_np_merge(
            block_path_debug,
            out_prefix,
            block_Z,
            block_X,
            block_Y,
            blocks_dir_np_merge,
            block_n_depth,
            block_n_rows,
            block_n_cols)

    

if '__main__' == __name__:
    try:
        (prog_name, 
         blocks_dir_data,
         blocks_dir_probs,
         blocks_dir_probs_ws,
         blocks_dir_probs_np,
         blocks_dir_ws,
         blocks_dir_np,
         blocks_dir_np_merge,
         blocks_dir_debug,
         out_prefix,
         block_Z, 
         block_X, 
         block_Y, 
         block_n_depth,
         block_n_rows,
         block_n_cols,
         cnn_patch_leg) = sys.argv[:17]
        
        block_Z = int(block_Z)
        block_X = int(block_X)
        block_Y = int(block_Y)
        
        block_n_depth = int(block_n_depth)
        block_n_rows = int(block_n_rows)
        block_n_cols = int(block_n_cols)
        
        cnn_patch_leg = int(cnn_patch_leg)
        
    except ValueError, e:
        sys.exit('USAGE: %s blocks_dir_data \
                            blocks_dir_probs, \
                            blocks_dir_probs_ws, \
                            blocks_dir_probs_np, \
                            blocks_dir_ws, \
                            blocks_dir_np, \
                            blocks_dir_np_merge, \
                            blocks_dir_debug, \
                            out_prefix \
                            [block_Z] \
                            [block_X] \
                            [block_Y] \
                            [block_n_depth] \
                            [block_n_rows] \
                            [block_n_cols] \
                            [cnn_patch_leg] ' % (sys.argv[0],))

    execute(
        blocks_dir_data,
        blocks_dir_probs,
        blocks_dir_probs_ws,
        blocks_dir_probs_np,
        blocks_dir_ws,
        blocks_dir_np,
        blocks_dir_np_merge,
        blocks_dir_debug,
        out_prefix,
        block_Z,
        block_X,
        block_Y,
        block_n_depth,
        block_n_rows,
        block_n_cols,
        cnn_patch_leg)

