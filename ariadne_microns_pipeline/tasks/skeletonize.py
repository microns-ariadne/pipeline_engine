import h5py
import json
import luigi
import numpy as np
import os
import subprocess
import shutil
import tempfile

import rh_logger
import rh_config
from microns_skeletonization import skeletonize, write_swc

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
    
    xy_nm = luigi.FloatParameter(
        default=4.0,
        description="The dimension of a voxel in the x and y directions")
    z_nm = luigi.FloatParameter(
        default=30.0,
        description="The dimension of a voxel in the z direction")
    decimation_factor = luigi.FloatParameter(
        default=0.5,
        description="Prune leaves if they are less than this fraction of "
                    "their parents' volume")
    
    def ariadne_run(self):
        seg = self.input().next().imread()
        result = skeletonize(seg, self.xy_nm, self.z_nm, self.decimation_factor,
                    self.cpu_count)
        if not os.path.isdir(self.skeleton_location):
            os.makedirs(self.skeleton_location)
        paths = []
        for label in result:
            path = os.path.join(self.skeleton_location, "%d.swc" % label)
            write_swc(path, result[label], self.xy_nm, self.z_nm)
            paths.append(path)
        with self.output().open("w") as fd:
            json.dump(paths, fd)

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

