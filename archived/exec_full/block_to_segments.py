
import os
import sys
import time

import cv2
import numpy as np

TYPE_NORMAL = 'normal'
TYPE_COMPLEX = 'complex'

def get_im_type(meta_im, white_im):
    
    type_id = TYPE_NORMAL
    
    mask_whites = (meta_im == white_im).astype(np.uint8)
    
    n_whites = mask_whites.sum()    
    n_pixels = white_im.shape[0] * white_im.shape[1]
    
    if n_whites > ((n_pixels) / 2):
        type_id = TYPE_COMPLEX
    
    return type_id
    

def get_processed_im(im_id, im_type, orig_block_images, meta_block_images):
    orig_im = orig_block_images[im_id]
    meta_im = meta_block_images[im_id]
    
    if im_type == TYPE_COMPLEX:
        res_im = orig_im
        
    else:
        mask_im = meta_im != 255
        res_im = orig_im * mask_im
    
    return res_im
    

def get_filenames(input_dir):
    filenames = os.listdir(input_dir)
    
    res_filenames = []
    for filename in filenames:
        filepath = os.path.join(input_dir, filename)
        
        if os.path.isdir(filepath):
            continue
        
        if filename.find('.txt') != -1:
            continue
        
        res_filenames.append(filename)
    
    res_filenames.sort()
    
    return res_filenames
    
def execute_ignore(block_id, orig_block_input_dir, block_output_dir):
    
    orig_block_filenames = get_filenames(orig_block_input_dir)
    
    orig_block_images = []
    print ' - ORIG DIR: %s' % (orig_block_input_dir,)
    for i, filename in enumerate(orig_block_filenames):
        print ' -- Read orig[%d]: %s' % (i, filename,)
        im = cv2.imread(os.path.join(orig_block_input_dir, filename), cv2.IMREAD_UNCHANGED)
        
        assert (im.dtype == np.uint8)
        
        orig_block_images.append(im)        
    
    im_shape = orig_block_images[0].shape
    
    n_images = len(orig_block_images)
    
    block_segments = []
    cur_block_type = None
    cur_block_images = []
    
    block_segments.append((TYPE_NORMAL, [im_id for im_id in xrange(n_images)]))
    
    n_segments = len(block_segments)
    
    for seg_id in xrange(n_segments):
        cur_block_type = block_segments[seg_id][0]
        cur_block_images = block_segments[seg_id][1]
        
        seg_name = ('block_id_%s_segment_%.4d_range_%.4d_%.4d_type_%s' % (block_id,
                                                                          seg_id,
                                                                          cur_block_images[0],
                                                                          cur_block_images[-1],
                                                                          cur_block_type))
        seg_path = os.path.join(block_output_dir, seg_name)
        
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)
        
        print ' == Generate segment %d [n_images = %d]' % (seg_id, len(cur_block_images))
        
        for im_id in cur_block_images:
            filename = os.path.basename(orig_block_filenames[im_id])
            seg_filename = '%s_seg_%d.png' % (filename, seg_id)
            seg_im_filepath = os.path.join(seg_path, seg_filename) 
            
            print ' -- Process image %d' % (im_id,)    
            processed_im = orig_block_images[im_id]
            
            assert (processed_im.dtype == np.uint8)
            
            print ' -- Write %s' % (seg_im_filepath,)    
            cv2.imwrite(seg_im_filepath, processed_im)
        
    
def execute_normal(block_id, orig_block_input_dir, meta_block_input_dir, block_output_dir):
    
    orig_block_filenames = get_filenames(orig_block_input_dir)
    meta_block_filenames = get_filenames(meta_block_input_dir)
    
    if orig_block_filenames != meta_block_filenames:
        raise Exception('ERROR: orig/meta dirs do not match:\n orig = %s\n meta = %s\n' % 
                        (orig_block_input_dir,
                         meta_block_input_dir))
    
    orig_block_images = []
    print ' - ORIG DIR: %s' % (orig_block_input_dir,)
    for i, filename in enumerate(orig_block_filenames):
        print ' -- Read orig[%d]: %s' % (i, filename,)
        im = cv2.imread(os.path.join(orig_block_input_dir, filename), cv2.IMREAD_UNCHANGED)
        
        assert (im.dtype == np.uint8)
        
        orig_block_images.append(im)        
    
    meta_block_images = []
    print ' - META DIR: %s' % (meta_block_input_dir,)
    for i, filename in enumerate(meta_block_filenames):
        print ' -- Read meta[%d]: %s' % (i, filename,)
        im = cv2.imread(os.path.join(meta_block_input_dir, filename), cv2.IMREAD_UNCHANGED)
        
        assert (im.dtype == np.uint8)
        
        meta_block_images.append(im)
    
    im_shape = orig_block_images[0].shape
    
    white_im = np.ones((im_shape), dtype = np.uint8) * 255
    
    n_images = len(meta_block_images)
    
    for i in xrange(n_images):
        assert (orig_block_images[i].shape == meta_block_images[i].shape)
    
    block_segments = []
    cur_block_type = None
    cur_block_images = []
    
    for im_id in xrange(n_images):
        orig_im = orig_block_images[im_id]
        meta_im = meta_block_images[im_id]
                
        if cur_block_type == None:
            cur_block_type = get_im_type(meta_im, white_im)
                
        cur_im_type = get_im_type(meta_im, white_im) 
            
        if cur_im_type == cur_block_type:
            cur_block_images.append(im_id)
        else:
            block_segments.append((cur_block_type, cur_block_images))
            cur_block_type = cur_im_type
            cur_block_images = [im_id,]
        
        block_segments.append((cur_block_type, cur_block_images))
    
    n_segments = len(block_segments)
    
    for seg_id in xrange(n_segments):
        cur_block_type = block_segments[seg_id][0]
        cur_block_images = block_segments[seg_id][1]
        
        seg_name = ('block_id_%s_segment_%.4d_range_%.4d_%.4d_type_%s' % (block_id,
                                                          seg_id,
                                                          cur_block_images[0],
                                                          cur_block_images[-1],
                                                          cur_block_type))
        seg_path = os.path.join(block_output_dir, seg_name)
        
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)
        
        print ' == Generate segment %d [n_images = %d]' % (seg_id, len(cur_block_images))
            
        for im_id in cur_block_images:
            filename = os.path.basename(orig_block_filenames[im_id])
            seg_filename = '%s_seg_%d.png' % (filename, seg_id)
            seg_im_filepath = os.path.join(seg_path, seg_filename) 
            
            print ' -- Process image %d' % (im_id,)    
            processed_im = get_processed_im(im_id, cur_block_type, orig_block_images, meta_block_images)
            
            assert (processed_im.dtype == np.uint8)
            
            print ' -- Write %s' % (seg_im_filepath,)    
            cv2.imwrite(seg_im_filepath, processed_im)
    
    print ' == Copy meta images'
    
    meta_path = os.path.join(block_output_dir, 'meta-images')    
    if not os.path.exists(meta_path):
        os.makedirs(meta_path)
        
    for i, meta_im in enumerate(meta_block_images):
        filename = os.path.basename(meta_block_filenames[i])
        filepath = os.path.join(meta_path, filename)
        
        print ' -- Write %s' % (filepath,)
        cv2.imwrite(filepath, meta_im)

    
if '__main__' == __name__:
    try:
        (prog_name, 
         is_ignore, 
         block_id, 
         orig_block_input_dir, 
         meta_block_input_dir, 
         block_output_dir = sys.argv[:6]
        
        is_ignore = int(is_ignore)
                
    except ValueError, e:
        sys.exit('USAGE: %s \
    [is_ignore] \
    [block_id] \
    [orig_block_input_dir] \
    [meta_block_input_dir] \
    [block_output_dir] ' % (sys.argv[0],))
    
    start_time_secs = time.time()
    
    if is_ignore:
        execute_ignore(block_id, orig_block_input_dir, block_output_dir)
    else:
        execute_normal(block_id, orig_block_input_dir, meta_block_input_dir, block_output_dir)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ELAPSED TIME: %s seconds' % (elapsed_time_secs,)
    

