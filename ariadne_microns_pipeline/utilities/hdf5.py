'''Standard utilities for handling HDF5 files

'''
import h5py
import contextlib
import os

@contextlib.contextmanager
def hdf5open(path, mode, cache):
    '''Open an HDF5 file with adjustable amount of cache
    
    :param path: path to the HDF5
    :param mode: open mode for the file
    :param cache: amount of cache to reserve for file in bytes
    '''
    #
    # From http://stackoverflow.com/questions/14653259/how-to-set-cache-settings-while-using-h5py-high-level-interface
    # Thank you unutbu
    #
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[2] = cache
    propfaid.set_cache(*settings)
    with contextlib.closing(h5py.h5f.open(path, fapl=propfaid)) as fid:
        yield h5py.File(fid, mode=mode)