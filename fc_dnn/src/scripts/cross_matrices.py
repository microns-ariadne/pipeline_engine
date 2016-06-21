
import sys
import os

def extract_header(header_line):
    n_matrices = int(header_line.split('total-matrices: ')[1])
    
    return n_matrices
    

def execute(output_filename, input_filenames):
    
    n_matrices_total = 0
    n_sizes = []
    files_data = []
    for input_filename in input_filenames:
        print 'Reading %s' % (input_filename,)
        
        f = open(input_filename, 'rb')  
        
        data_matrices = f.read().split('matrix-start:')
        
        f.close()
        
        n_matrices = extract_header(data_matrices[0])
        
        files_data.append(data_matrices)
        n_sizes.append(n_matrices)
        
        n_matrices_total += n_matrices
    
    for i in xrange(len(n_sizes)):
        if n_sizes[0] != n_sizes[i]:
            raise Exception('All files must have the same number of matrices.')
    
    res_data = 'total-matrices: %d\n' % (n_matrices_total,)
    
    out_m_id = 1
    for iter_id in xrange(n_sizes[0]):
        m_id = iter_id + 1
        
        for f_id in xrange(len(input_filenames)):
            matrix_data = files_data[f_id][m_id]
            
            matrix_data = matrix_data.replace('[%d]' % (m_id,), '[%d]' % (out_m_id,))
            
            matrix_data = 'matrix-start:' + matrix_data
            
            res_data += matrix_data
            
            out_m_id += 1
    
    f = open(output_filename, 'wb')
    f.write(res_data)
    f.close()
    print 'Generated %s for %d matrices' % (output_filename, n_matrices_total)
    
    
if '__main__' == __name__:
    try:
        prog_name, output_filename = sys.argv[:2]
        input_filenames = sys.argv[2:]

    except ValueError, e:
        sys.exit('USAGE: %s [output_filename] [[input_filenames]]' % (sys.argv[0],))

    execute(output_filename, input_filenames)
