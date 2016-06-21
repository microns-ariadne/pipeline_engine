
import sys
import os
import struct
import numpy as np
from imread import imread

MAGIC_TOTAL_MATRIX = 0x1212121221212121
MAGIC_MATRIX_START = 0x1111111111111111
MAGIC_MATRIX_END = 0x2222222222222222

def generate_bin_data(n_matrices, d1, d2, d3, matrices):
    
    bin_data = []
    
    bin_data.append(struct.pack('L', MAGIC_TOTAL_MATRIX))
    bin_data.append(struct.pack('LLLL', n_matrices, d1, d2, d3))
    
    for m_id in xrange(n_matrices):
        matrix = matrices[m_id]
        
        bin_data.append(struct.pack('L', MAGIC_MATRIX_START))
        
        values = []
        for cur_d1 in xrange(d1):     
            for cur_d2 in xrange(d2):
                for cur_d3 in xrange(d3):
                    values.append(matrix[cur_d1,cur_d2,cur_d3])
                    
        bin_data.append(struct.pack('B' * len(values), *values))
                    
        bin_data.append(struct.pack('L', MAGIC_MATRIX_END))

    bin_data = ''.join(bin_data)
    
    return bin_data
    

def execute(is_3D_bins, is_stacked, input_dir, output_dir, output_prefix):
    
    filenames = os.listdir(input_dir)
    
    print 'Reading %d images from %s' % (len(filenames), input_dir)
    
    images = []
    
    if is_3D_bins:
        
        tmp_images = []
        for filename in filenames:
            tmp_images.append(imread(os.path.join(input_dir, filename)))
        
        for i in xrange(len(filenames)-2):
            image_1 = tmp_images[i]
            image_2 = tmp_images[i+1]
            image_3 = tmp_images[i+2]
            
            image_3D = np.array([image_1, image_2, image_3])
            image_3D = image_3D.transpose((1,2,0))
            
            images.append(image_3D)
            
            print 'construct 3D image %d of shape %r' % (i+1, image_3D.shape)
        
    else:
        for filename in filenames:
            images.append(imread(os.path.join(input_dir, filename)))
    
    d1 = images[0].shape[0]
    d2 = images[0].shape[1]
    d3 = images[0].shape[2]
    
    f_images = []
    for im in images:
        
        if (d1 != im.shape[0]) or (d2 != im.shape[1]) or (d3 != im.shape[2]):
           raise Exception('All images must be of the same size [%d,%d,%d]' % (d1, d2, d3))
        
        if is_3D_bins:
            f_images.append(im)
        else:
            im = np.array(im, ndmin=3)
            im = im.transpose((1,2,0))
            f_images.append(im)
        
    if is_stacked:
        
        bin_data = generate_bin_data(len(f_images), d1, d2, d3, f_images)
        
        output_filename = os.path.join(output_dir, '%s_stacked.bin' % (output_prefix,))
        f = open(output_filename, 'wb')
        f.write(bin_data)
        f.close()
        print 'Generated %s for %d matrices of shape [%d,%d,%d]' % (output_filename, len(f_images), d1, d2, d3)
    
    else:
        
        for i, f_image in enumerate(f_images):
            bin_data = generate_bin_data(1, d1, d2, d3, [f_image])
            
            if is_3D_bins:
                image_idx = i + 1
            else:
                image_idx = i
            
            output_filename = os.path.join(output_dir, '%s_%d.bin' % (output_prefix, image_idx))
            f = open(output_filename, 'wb')
            f.write(bin_data)
            f.close()
            print 'Generated %s for %d matrices of shape [%d,%d,%d]' % (output_filename, 1, d1, d2, d3)
    
    
if '__main__' == __name__:
    try:
        prog_name, is_3D_bins, is_stacked, input_filename, output_dir, output_prefix = sys.argv[:6]
        
        is_stacked = int(is_stacked)
                 
    except ValueError, e:
        sys.exit('USAGE: %s [is_3D_bins] [is_stacked] [input_dir] [output_dir] [output_prefix]' % (sys.argv[0],))


    execute(is_3D_bins, is_stacked, input_filename, output_dir, output_prefix)

