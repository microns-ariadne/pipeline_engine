
import sys
import os
import struct
import numpy as np
from imread import imread, imsave

MAGIC_TOTAL_MATRIX = 0x1212121221212121
MAGIC_MATRIX_START = 0x1111111111111111
MAGIC_MATRIX_END = 0x2222222222222222

LONG_SIZE = 8
BYTE_SIZE = 1

def extract_matrices(input_filename):
    
    f = open(input_filename, 'rb')
    
    bin_data = f.read()
    
    f.close()
    
    cur_idx = 0
    
    magic_total_matrix = struct.unpack('L', bin_data[cur_idx : cur_idx + LONG_SIZE])[0]
    cur_idx += LONG_SIZE
        
    if magic_total_matrix != MAGIC_TOTAL_MATRIX:
        raise Exception('Unexpected magic_total_matrix [%ld] (expected: [%ld])' % (magic_total_matrix, MAGIC_TOTAL_MATRIX))
    

    
    (n_matrices, d1, d2, d3) = struct.unpack('LLLL', bin_data[cur_idx : cur_idx + LONG_SIZE * 4])
    cur_idx += LONG_SIZE * 4
    
    matrices = []
    for m_id in xrange(n_matrices):
        matrix = np.zeros((d1,d2,d3), dtype=np.uint8)
        matrices.append(matrix)
            
        magic_matrix_start = struct.unpack('L', bin_data[cur_idx : cur_idx + LONG_SIZE])[0]
        cur_idx += LONG_SIZE
        
        if magic_matrix_start != MAGIC_MATRIX_START:
            raise Exception('Unexpected magic_matrix_start [%ld] (expected: [%ld])' % (magic_matrix_start, MAGIC_MATRIX_START))
            
        
        for cur_d1 in xrange(d1):
            for cur_d2 in xrange(d2):
                for cur_d3 in xrange(d3):
                    byte = struct.unpack('B', bin_data[cur_idx : cur_idx + BYTE_SIZE])[0]
                    cur_idx += BYTE_SIZE
                    
                    matrix[cur_d1,cur_d2,cur_d3] = 255 - byte
                    
        magic_matrix_end = struct.unpack('L', bin_data[cur_idx : cur_idx + LONG_SIZE])[0]
        cur_idx += LONG_SIZE

        if magic_matrix_end != MAGIC_MATRIX_END:
            raise Exception('Unexpected magic_matrix_end [%ld] (expected: [%ld])' % (magic_matrix_end, MAGIC_MATRIX_END))
    
    return matrices
    
      
def execute(input_filename, output_filename_prefix):
    
    print 'Reading %s' % (input_filename,)
    
    matrices = extract_matrices(input_filename)
    
    print 'Extraced %d matrices of shape %r' % (len(matrices), matrices[0].shape)
    
    for i in xrange(len(matrices)):
        matrix = matrices[i][:,:,0]
        
        output_filename = '%s_%d.tif' % (output_filename_prefix, i+1)
        imsave(output_filename, matrix)
        
        print 'Generated %s for matrix with shape %r' % (output_filename, matrix.shape)
    

    
if '__main__' == __name__:
    try:
        prog_name, input_filename, output_filename_prefix = sys.argv[:3]
                
    except ValueError, e:
        sys.exit('USAGE: %s [input_filename] [output_filename_prefix]' % (sys.argv[0],))


    execute(input_filename, output_filename_prefix)

