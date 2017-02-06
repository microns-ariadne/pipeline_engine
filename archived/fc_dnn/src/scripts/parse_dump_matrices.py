
import sys
import os
import numpy as np

def extract_header(header_line):
    parts = header_line.split('of shape')
    
    n_matrices = int(parts[0].split('=')[1])
    
    dims = [int(x) for x in parts[1].strip('[ ]').split(',')]
    
    d1 = dims[0]
    d2 = dims[1]
    d3 = dims[2]
    
    return (n_matrices, d1, d2, d3)
    
def extract_matrix_header(m_header_line):
    parts = m_header_line.split('aux_data =')
    
    m_id = int(parts[0].split('MATRIX')[1].strip('[ ]'))
    
    aux_data = float(parts[1])
    
    return (m_id, aux_data)

def extract_matrix(data_lines, d1, d2, d3):
    matrix = []
    
    for i in xrange(d1):
        parts = data_lines[i].split('[%d] ' % (i,))
                
        entries = [ x.strip(' )') for x in parts[1].split('( ')[1:]]
        
        row = []
        for j in xrange(d2):
            
            nums = [float(x) for x in entries[j].split(' ')]
            
            if len(nums) != d3:
                raise Exception('Unexpected d3 depth: %d (expected: %d)' % (len(nums), d3))
            
            row.append(nums)
        
        matrix.append(row)
    
    return np.array(matrix)
    
      
def parse_dump_matrices(input_filename):
    
    print 'Reading %s' % (input_filename,)
    
    f = open(input_filename, 'rb')
    
    dump_data_lines = f.read().splitlines();
    
    f.close()
    
    dump_data_lines = dump_data_lines[1:]
    
    (n_matrices, d1, d2, d3) = extract_header(dump_data_lines[0])
    
    print 'Extracting %d matrices of shape [%d,%d,%d]' % (n_matrices, d1, d2, d3)
    
    dump_data_lines = dump_data_lines[2:]
    
    matrices = []
    aux_data = []
    for i in xrange(n_matrices):
        dump_data_lines = dump_data_lines[1:]
        
        (in_m_id, in_aux_data) = extract_matrix_header(dump_data_lines[0])
        
        print ' -- matrix %d with aux_data = %f' % (in_m_id, in_aux_data)
        
        if (in_m_id != (i+1)):
            raise Exception('Unexpected matrix id [%d] (expected: %d)' % (in_m_id, i+1))
        
        aux_data.append(in_aux_data)
        
        dump_data_lines = dump_data_lines[2:]
        
        matrices.append(extract_matrix(dump_data_lines, d1, d2, d3))
        
        dump_data_lines = dump_data_lines[d1:]
        
    res_matrices = np.array(matrices)
    res_aux_data = np.array(aux_data)
    
    print 'Resulting numpy matrices shape %r' % (res_matrices.shape,)
    print 'Resulting numpy aux_data shape %r' % (res_aux_data.shape,)
    
    return (res_matrices, res_aux_data)
    

