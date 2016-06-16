import os
import string

#import h5py
import numpy as np

from util import *

# def compress_h5(filepath, dset):
#     print 'compress_h5: start'
#     print '  -- filepath : %s' % (filepath,)
#     print '  -- dset     : %s' % (dset,)
# 
#     h5_file = h5py.File(filepath, 'r')
#     assert(len(h5_file.keys()) == 1)
#     dset_data = h5_file[dset]
#     
#     tmp_h5_filepath = filepath + '_temp_gzip'
# 
#     tmp_h5_file = h5py.File(tmp_h5_filepath, 'w')
#     tmp_h5_file.create_dataset(dset, data = dset_data, compression = 'gzip')
# 
#     tmp_h5_file.close()
#     h5_file.close()
# 
#     os.unlink(filepath)
#     os.rename(tmp_h5_filepath, filepath)
#     
#     print 'compress_h5: finish'
    
def slice_data(data1, data2, width, axis):
    ndim = len(data1.shape)
    assert ndim == len(data2.shape)
    assert data1.dtype == data2.dtype

    slice_shape = [ ]

    for i in xrange(ndim):
        if i != axis:
            assert data1.shape[i] == data2.shape[i]
            slice_shape.append(data1.shape[i])
        else:
            slice_shape.append(width + min(width, data2.shape[i]))

    slice_data = np.zeros(slice_shape, dtype=data1.dtype)
    
    slice_coord = [ slice(None) ] * ndim
    slice_coord[axis] = slice(0, width)

    data_coord = [ slice(None) ] * ndim
    data_coord[axis] = slice(-width, None)

    slice_data[slice_coord] = data1[data_coord]

    slice_coord[axis] = slice(width, None)

    data_coord[axis] = slice(None, width)

    slice_data[slice_coord] = data2[data_coord]

    return slice_data

