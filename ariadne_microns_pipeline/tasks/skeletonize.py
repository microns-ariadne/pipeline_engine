import h5py
import luigi
import numpy as np
import os
import subprocess
import tempfile

import rh_logging
import rh_config

from ..parameters import VolumeParameter, DatasetLocationParameter
from ..targets.factory import TargetFactory
from .utilities import RequiresMixin, RunMixin


class SkeletonizeTaskMixin:
    
    volume=VolumeParameter(
        description="The volume to skeletonize, in global coordinates")
    segmentation_location=DatasetLocationParameter(
        description="The location of the input segmentation")
    skeleton_location=luigi.Parameter(
        description="The location for the skeleton")
    
    def input(self):
        yield TargetFactory().get_volume_target(
            location=self.segmentation_location,
            volume=self.volume)
    
    def output(self):
        return luigi.LocalTarget(self.skeleton_location)

class SkeletonizeRunMixin:
    
    downsampling_scale=luigi.IntParameter(
        default=1,
        description="Factor to downsample the input. "
        "Valid choices are 1, 2, and 4")
    
    def copy_to_hdf5(self):
        '''Copy the input segmentation to a local HDF5 file'''
        
        seg = self.input().next().imread()
        self.hdf5_fd, self.hdf5_file = tempfile.mkstemp(suffix=".h5")
        with h5py.File(self.hdf5_file, "w") as h5:
            assert isinstance(h5, h5py.Group)
            h5.create_dataset("stack", data=seg, dtype=np.uint32)
    
    def post_run(self):
        '''Delete the temporary .h5 file containing the segmentation'''
        os.close(self.hdf5_fd)
        os.remove(self.hdf5_file)
    
    def ariadne_run(self):
        self.copy_to_hdf5()
        try:
            config = rh_config.config["skeletonization"]
            home_dir = config["home-dir"]
            ld_library_path = os.pathsep.join(config["ld_library_path"])
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = ld_library_path
            args = ["main", "-sb", 
                    self.downsampling_scale,
                    self.hdf5_file, self.output().path]
            subprocess.check_call(args, cwd=home_dir, env=env)
        finally:
            self.post_run()

