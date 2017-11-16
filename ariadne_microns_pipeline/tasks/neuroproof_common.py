'''Common routines and constants for neuroproof learning and prediction

'''

import enum
import h5py
import numpy as np
import os
from scipy.ndimage import grey_dilation
import tempfile
import tifffile

from ..targets import DestVolumeReader
from ..targets.volume_target import write_simple_loading_plan

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
    '''
    Run using https://github.com/timkaler/fast_neuroproof
    '''
    FAST = 4

def write_seg_volume(watershed_path, seg_target, dataset_name):
    '''Write the watershed out to an hdf5 file for Neuroproof
    
    :param watershed_path: the path to the HDF5 file to write
    :param seg_target: the volume to write
    :param dataset_name: the HDF5 dataset's key name
    '''
    with h5py.File(watershed_path, "w") as fd:
        seg_volume = seg_target.imread().astype(np.uint32)
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
    as the second channel. If False, invert it. If None, don't use one.
    '''
    prob_volume = [prob_target.imread().astype(np.float32) / 255.]
    if duplicate is not None:
        prob_volume.append(prob_volume[0] if duplicate else 1-prob_volume[0])
    for tgt in additional_map_targets:
        prob_volume.append(tgt.imread().astype(np.float32) / 255.)
    prob_volume = np.array(prob_volume)
    if transpose:
        prob_volume = prob_volume.transpose(3, 2, 1, 0)
    else:
        prob_volume = prob_volume.transpose(1, 2, 3, 0)
    with h5py.File(pred_path, "w") as fd:
        fd.create_dataset(dataset_name, data=prob_volume)

class NeuroproofDilateMixin:
    '''A mixin that creates a dilated version of the membrane prediction
    
    Assumes that your mixin class has the following luigi parameters:
    
    self.dilation_xy - size of structuring element in x/y direction
    self.dilation_z - size of structuring element in the Z direction
    self.prob_loading_plan_path - location of the membrane probabilities
    
    '''
    def dilate_membrane_prediction(self):
        '''Create a dilated copy of the membrane on disk
        
        :returns: loading plan path of the copy
        '''
        if self.dilation_xy == 1 and self.dilation_z == 1:
            return self.prob_loading_plan_path
        membrane_tgt = DestVolumeReader(self.prob_loading_plan_path)
        strel = np.ones((self.dilation_z, self.dilation_xy, self.dilation_xy),
                        bool)
        membrane = grey_dilation(membrane_tgt.imread(), footprint=strel)
        tif_fp, tif_path = tempfile.mkstemp(".tif")
        tifffile.imsave(tif_path, membrane, compress=3)
        os.close(tif_fp)
        loading_plan_fp, loading_plan_path = tempfile.mkstemp(".loading.plan")
        write_simple_loading_plan(loading_plan_path, 
                                  tif_path, membrane_tgt.volume, 
                                  "dilated", 
                                  np.dtype(membrane_tgt.dtype).name)
        os.close(loading_plan_fp)
        return loading_plan_path
    
    def delete_dilated_loading_plan(self, loading_plan):
        '''Delete the loading plan and tif file from dilate_membrane_prediction
        
        :param loading_plan: the loading plan path returned by 
        dialte_membrane_prediction
        '''
        if self.prob_loading_plan_path == loading_plan:
            return
        tgt = DestVolumeReader(loading_plan)
        for tif_path, volume in tgt.get_tif_paths_and_volumes():
            os.remove(tif_path)
        os.remove(loading_plan)
        
    


