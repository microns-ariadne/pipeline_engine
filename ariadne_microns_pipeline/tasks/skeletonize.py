import h5py
import luigi
import numpy as np
import os
import subprocess
import shutil
import tempfile

import rh_logger
import rh_config

from ..parameters import VolumeParameter, DatasetLocationParameter
from ..targets.factory import TargetFactory
from .utilities import RequiresMixin, RunMixin, CILKCPUMixin


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
        return luigi.LocalTarget(self.skeleton_location+".done")

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
    
    def rewrite_swc(self, path):
        '''Rewrite a .swc file so that the nodes are numbered starting at 1
        
        :param path: the path to the .swc file
        '''
        node_map = {}
        node_idx = 1
        nodes = []
        with open(path, "r") as fd:
            for line in fd:
                node, ntype, x, y, z, r, conn = line.strip().split(" ")
                node_map[node] = node_idx
                if float(r) == 0:
                    r = "1.000000"
                node_idx += 1
                nodes.append((node, ntype, x, y, z, r, conn))
        new_path = path + ".rewrite"
        with open(new_path, "w") as fd:
            for node, ntype, x, y, z, r, conn in nodes:
                x = float(x) + self.volume.x
                y = float(y) + self.volume.y
                z = float(z) + self.volume.z
                node = node_map[node]
                conn = node_map.get(conn, int(conn))
                fd.write("%d %s %.6f %.6f %.6f %s %d\n" % (
                    node, ntype, x, y, z, r, conn))
        os.rename(path, path+".old")
        os.rename(new_path, path)
        os.remove(path+".old")
    
    def ariadne_run(self):
        self.copy_to_hdf5()
        try:
            if not os.path.isdir(self.skeleton_location):
                os.mkdir(self.skeleton_location)
            swcdir = os.path.join(self.skeleton_location, "SWC")
            if not os.path.isdir(swcdir):
                os.mkdir(swcdir)
            config = rh_config.config["skeletonization"]
            home_dir = config["home-dir"]
            ld_library_path = os.pathsep.join(config["ld_library_path"])
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = ld_library_path
            self.configure_env(env)
            args = [os.path.join(home_dir, "main"),
                    "-sb", 
                    str(self.downsampling_scale),
                    self.hdf5_file, self.skeleton_location,
                    str(self.volume.width),
                    str(self.volume.height),
                    str(self.volume.depth)]
            #args = '"'+'" "'.join(args)+'"'
            subprocess.check_call(args, cwd=home_dir, env=env)
            for filename in os.listdir(swcdir):
                if filename.endswith(".swc"):
                    self.rewrite_swc(os.path.join(swcdir, filename))
            with self.output().open("w") as fd:
                fd.write("Done")
        finally:
            self.post_run()

class SkeletonizeTask(SkeletonizeTaskMixin,
                      SkeletonizeRunMixin,
                      RequiresMixin,
                      RunMixin,
                      CILKCPUMixin,
                      luigi.Task):
    '''Skeletonize a segmentation volume
    
    This task takes a 3d segmentation volume and performs a skeletonization
    saving the results in an SWC file.
    
    This code depends on the binary skeletonization executable. This should
    be produced by the top-level makefile which compiles the program,
    skeletonization/graph_extraction/main, and a lookup table that drives
    the erosion process, skeletonization/graph_extraction/LUT/LUT.txt. See
    the README.md file for the parameters for .rh-config.yaml that are
    relevant to this program.
    '''
    task_namespace="ariadne_microns_pipeline"

