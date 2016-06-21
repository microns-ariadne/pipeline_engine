
import sys
import os
import numpy as np
import scipy.io as sio
from parse_dump_matrices import parse_dump_matrices
       
def execute(input_filename, output_filename_prefix):
    
    (res_matrices, res_aux_data) = parse_dump_matrices(input_filename)
    
    mat_output_matrices = '%s_matrices.mat' % (output_filename_prefix,)
    mat_output_aux_data = '%s_aux_data.mat' % (output_filename_prefix,)
    
    sio.savemat(mat_output_matrices, {'matrices' : res_matrices })
    print 'Generated %s' % (mat_output_matrices,)
    
    sio.savemat(mat_output_aux_data, {'aux_data' : res_aux_data })
    print 'Generated %s' % (mat_output_aux_data,)
    

    
if '__main__' == __name__:
    try:
        prog_name, input_filename, output_filename_prefix = sys.argv[:3]
                
    except ValueError, e:
        sys.exit('USAGE: %s [input_filename] [output_filename_prefix]' % (sys.argv[0],))


    execute(input_filename, output_filename_prefix)

