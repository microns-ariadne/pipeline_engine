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
    use_neutu = luigi.BoolParameter(
        description="Use NeuTu's skeletonizer")
    downsampling_factor = luigi.IntParameter(
        default=0,
        description="Downsample image by this factor to reduce processing time")
    minimal_length = luigi.FloatParameter(
        default=50.0,
        description="Neutu: minimum length of a branch")
    keep_single_object = luigi.BoolParameter(
        description="Neutu: keep an object even if it would otherwise be filtered by size")
    rebase = luigi.BoolParameter(
        description="Neutu: reset the starting point to a terminal point")
    filling_hole = luigi.BoolParameter(
        description="Neutu: Fill holes in the object before skeletonizing")
    maximal_distance = luigi.FloatParameter(
        default=-1.0,
        description="Neutu: maximum distance to connect isolated branches")
    minimal_object_size = luigi.IntParameter(
        default=0,
        description="Neutu: minimum area for an object before removing.")
    skeletonize_stack_location = luigi.Parameter(
        default="/dev/null",
        description="Location of the NeuTu skeletonize_stack binary")
    
    def ariadne_run(self):
        if self.use_neutu:
            self.skeletonize_using_neutu()
        else:
            self.skeletonize_using_microns()
    
    def skeletonize_using_microns(self):
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
    
    def skeletonize_using_neutu(self):
        '''Skeletonize by running "skeletonize_stack" to execute NeuTu's code
        
        See https://github.com/vcg/microns_skeletonization
        '''
        #
        # skeletonize_stack takes a single JSON file that tells it what
        # to do. Here's the schema
        #
        # volume:
        #   x: x coordinate of the volume's origin
        #   y: y coordinate of the volume's origin
        #   z: z coordinate of the volume's origin
        #   width: width of the volume in voxels
        #   height: height of the volume in voxels
        #   depth: depth of the volume in voxels
        #   xy_nm: length of a voxel in the x and y directions in nanometers
        #   z_nm: length of a voxel in the z direction in nanometers
        #   planes: array of strings of the filenames of each plane.
        # processing_instructions:
        #   See ./neurolabi/json/skeletonize.schema.json in the NeuTu project
        # output:
        #   Dictionary of label to .swc file to write
        #
        seg_target = self.input().next()
        #
        # Find the labels we should write
        #
        seg = seg_target.imread()
        areas = np.bincount(seg.ravel())
        areas[0] = 0
        labels = np.where(areas > 0)[0]
        #
        # Build the input volume
        #
        volume = dict(x=self.volume.x,
                      y=self.volume.y,
                      z=self.volume.z,
                      width=self.volume.width,
                      height=self.volume.height,
                      depth=self.volume.depth,
                      xy_nm=self.xy_nm,
                      z_nm=self.z_nm,
                      planes=seg_target.get_filenames())
        #
        # Build the processing instructions
        #
        downsample_xy = self.downsampling_factor
        if downsample_xy == 1:
            # downsampleInterval=0 means "Don't downsample" which is the
            # intent of the user specifying "1"
            #
            downsample_xy = 0
        processing_instructions = dict(
            downsampleInterval=[downsample_xy, downsample_xy, 0],
            minimalLength=self.minimal_length,
            keepingSingleObject=self.keep_single_object,
            rebase=self.rebase,
            fillingHole=self.filling_hole,
            maximalDistance=self.maximal_distance,
            minimalObjectSize=self.minimal_object_size)
        #
        # Build the output
        #
        output_target = self.output()
        root = os.path.splitext(output_target.path)[0]
        if not os.path.isdir(root):
            os.makedirs(root)
        output = dict(
            [(label, os.path.join(root, "%d.swc" % label))
             for label in labels])
        #
        # Write the config file to temp storage.
        #
        d = dict(volume=volume,
                 processing_instructions=processing_instructions,
                 output=output)
        fileno, filename = tempfile.mkstemp(".json")
        try:
            fd = os.fdopen(fileno, "w")
            json.dump(d, fd)
            fd.close()
            #
            # Call the subprocess with the file
            #
            subprocess.check_call([self.skeletonize_stack_location,
                                   filename])
            with output_target.open("w") as fd:
                json.dump(output.values(), fd)
        finally:
            os.remove(filename)

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

