'''Common routines and constants for neuroproof learning and prediction

'''

import enum
import h5py
import numpy as np

class NeuroproofVersion(enum.Enum):
    '''The command-line convention for the neuroproof runtime
    '''
    '''
    Run using the .json instructions for retrieving files.
    '''
    MIT = 1
    '''
    Run using the conventions of Neuroproof-flyem's neuroproof_graph_predict
    '''
    FLY_EM = 2
    '''
    Run using the conventions of neuroproof_minimal's Neuroproof_stack
    '''
    MINIMAL = 3

def write_seg_volume(watershed_path, seg_target, dataset_name):
    '''Write the watershed out to an hdf5 file for Neuroproof
    
    :param watershed_path: the path to the HDF5 file to write
    :param seg_target: the volume to write
    :param dataset_name: the HDF5 dataset's key name
    '''
    with h5py.File(watershed_path, "w") as fd:
        seg_volume = seg_target.imread().astype(np.int32)
        fd.create_dataset(dataset_name, data=seg_volume)

def write_prob_volume(prob_target, additional_map_targets, pred_path, 
                      dataset_name, transpose=True, duplicate=False):
    '''Write Neuroproof's probabilities hdf file
    
    :param prob_target: the membrane probabilities volume
    :param additional_map_targets: a list of additional volumes to write
    out to the probabilities file.
    :param pred_path: the name of the HDF5 file to write
    :param dataset_name: the HDF5 dataset's key name
    :param transpose: True for Ilastik-style volumes (x, y, z, c) for
                      neuroproof_graph_learn. False for z, y, x, c volumes
                      for Neuroproof_stack.
    :param duplicate: if True, duplicate the first (membrane) channel
    as the second channel. If False, invert it.
    '''
    prob_volume = prob_target.imread().astype(np.float32) / 255.
    prob_volume = [prob_volume, 
                   prob_volume if duplicate else 1-prob_volume]
    for tgt in additional_map_targets:
        prob_volume.append(tgt.imread().astype(np.float32) / 255.)
    prob_volume = np.array(prob_volume)
    if transpose:
        prob_volume = prob_volume.transpose(3, 2, 1, 0)
    else:
        prob_volume = prob_volume.transpose(1, 2, 3, 0)
    with h5py.File(pred_path, "w") as fd:
        fd.create_dataset(dataset_name, data=prob_volume)


