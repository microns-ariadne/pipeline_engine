'''The BlockTask constructs an output block from input blocks

'''

import json
import luigi
import numpy as np
from scipy.ndimage import map_coordinates
from .connected_components import ConnectivityGraph
from .utilities import RequiresMixin, RunMixin
from ..targets.factory import TargetFactory, TFEnums
from ..parameters import DatasetLocationParameter, VolumeParameter
from ..parameters import MultiVolumeParameter

class BlockTaskMixin:
    '''The block task constructs creates a block of data from volumes'''

    output_location = DatasetLocationParameter(
        description="Location of volume to be created")
    output_volume = VolumeParameter(
        description="Volume to be extracted from input datasets")
    input_volumes = MultiVolumeParameter(
        description="The volumes that will be composited to form the output "
        "volume.")
    downsample_xy = luigi.FloatParameter(
        default=1.0,
        description="# of input voxels in the x and y direction that become "
        "one voxel in the output volume")
    downsample_z = luigi.FloatParameter(
        default=1.0,
        description="# of input voxels in the z direction that become one "
        "voxel in the output volume")
    wants_nearest = luigi.BoolParameter(
        description="Instead of interpolating, take the value of the nearest "
                    "pixel (e.g. for segmentation data)")
    mapping = luigi.Parameter(
        default="/dev/null",
        description="A local to global segmentation mapping produced by "
                    "the AllConnectedComponents task.")
    target_type = luigi.EnumParameter(
        enum=TFEnums,
        default=TFEnums.use_png_volume,
        description="The target tyle for the output volume, e.g. PNG or HDF5")
    xy_chunking = luigi.IntParameter(
        default=1024,
        description="The size of an HDF5 chunk in the x and y directions "
                    "(HDF5 only).")
    z_chunking = luigi.IntParameter(
        default=1,
        description="The size of an HDF5 chunk in the Z direction (HDF5 only).")
    
    def input(self):
        '''Return the volumes to be assembled'''
        tf = TargetFactory()
        for d in self.input_volumes:
            yield tf.get_volume_target(d["location"], d["volume"])
    
    def output(self):
        '''Return the volume target that will be written'''
        tf = TargetFactory()
        return tf.get_volume_target(self.output_location, self.output_volume,
                                    target_type=self.target_type)
    
    def estimate_memory_usage(self):
        '''Return an estimate of bytes of memory required by this task'''
        v1 = np.prod([1888, 1888, 100]) * 2
        m1 = 921211 * 1000
        v2 = np.prod([1888, 1888, 52]) * 2
        m2 = 601100 * 1000
        #
        # Model is Ax + B where x is volume in voxels of the largest input
        # volume + the output volume.
        #
        # The numbers should depend on the bit depth of the input which is
        # generally unknown but always = 8 in the pipeline (for now).
        #
        B = (v1 * m2 - v2 * m1) / (v1 - v2)
        A = (float(m1) - B) / v1
        v = np.prod([self.output_volume.width, 
                     self.output_volume.height, 
                     self.output_volume.depth]) +\
            np.max([np.prod([volume.width, volume.height, volume.depth])
                    for volume in self.input()])
        return int(A * v + B)


class BlockTaskRunMixin:
    '''Combine the inputs to produce the output
    
    The algorithm is simple - take the inputs in turn, find the intersection
    with the output volume. The output volume datatype is the same as that
    of the first input volume.
    '''
    
    def ariadne_run(self):
        '''Loop through the inputs, compositing their volumes on the output'''
        
        # TO_DO: a clever implementation would optimize the block organization
        #        of the output volume dataset by looking at the average
        #        size of the blocks being written.
        output_volume = self.output()
        first = True
        #
        # The downsampled output coordinates
        #
        x0d = self.output_volume.x
        x1d = self.output_volume.x1
        y0d = self.output_volume.y
        y1d = self.output_volume.y1
        z0d = self.output_volume.z
        z1d = self.output_volume.z1
        if self.mapping == "/dev/null":
            mapping = None
        else:
            mapping = ConnectivityGraph.load(open(self.mapping, "r"))
            
        for input_volume in self.input():
            #
            # Compute the portion of the input volume that overlaps
            # with the requested output volume.
            #
            x0 = int(max(x0d, 
                         np.floor(input_volume.volume.x / self.downsample_xy)))
            x1 = int(min(x1d, 
                         np.ceil(input_volume.volume.x1 / self.downsample_xy)))
            if x0 >= x1:
                continue
            y0 = int(max(y0d, 
                         np.floor(input_volume.volume.y / self.downsample_xy)))
            y1 = int(min(y1d, 
                         np.ceil(input_volume.volume.y1 / self.downsample_xy)))
            if y0 >= y1:
                continue
            z0 = int(max(z0d, 
                         np.floor(input_volume.volume.z / self.downsample_z)))
            z1 = int(min(z1d, 
                         np.ceil(input_volume.volume.z1 / self.downsample_z)))
            if z0 >= z1:
                continue
            if self.downsample_xy == 1.0 and self.downsample_z == 1.0:
                subvolume = input_volume.imread_part(
                    x0, y0, z0, x1-x0, y1-y0, z1-z0)
            else:
                x0i = x0 * self.downsample_xy
                x0ii = int(x0i)
                x1i = x1 * self.downsample_xy
                x1ii = int(np.ceil(x1i))
                y0i = y0 * self.downsample_xy
                y0ii = int(y0i)
                y1i = y1 * self.downsample_xy
                y1ii = int(np.ceil(y1i))
                z0i = z0 * self.downsample_z
                z0ii = int(z0)
                z1i = z1 * self.downsample_z
                z1ii = int(np.ceil(z1i))
                x = np.linspace(x0i, x1i, (x1 - x0), endpoint=False)\
                    [np.newaxis, np.newaxis, :]
                y = np.linspace(y0i, y1i, (y1 - y0), endpoint=False)\
                    [np.newaxis, :, np.newaxis]
                z = np.linspace(z0i, z1i, (z1 - z0), endpoint=False)\
                    [:, np.newaxis, np.newaxis]
                base = np.zeros(
                    (z.shape[0], y.shape[1], x.shape[2]), np.float32)
                coords = [base + (z  - z0ii).astype(np.float32),
                          base + (y - y0ii).astype(np.float32),
                          base + (x - x0ii).astype(np.float32)]
                subvolume = input_volume.imread_part(
                    x0ii, y0ii, z0ii, x1ii - x0ii, y1ii - y0ii, z1ii - z0ii)
                if self.wants_nearest:
                    order = 0
                else:
                    order = 3
                subvolume = map_coordinates(subvolume, coords, order=order)
            if mapping is not None:
                subvolume = mapping.convert(subvolume, input_volume.volume)
            if first:
                first = False
                if self.target_type == TFEnums.use_hdf5_volume:
                    chunks=(min(self.z_chunking, self.output_volume.depth),
                            min(self.xy_chunking, self.output_volume.height),
                            min(self.xy_chunking, self.output_volume.width))
                    output_volume.create_volume(
                        subvolume.dtype,
                        chunks=chunks,
                        compression="gzip")
                else:
                    output_volume.create_volume(subvolume.dtype)
            output_volume.imwrite_part(subvolume, x0, y0, z0)
                
        output_volume.finish_volume()


class BlockTask(BlockTaskMixin,
                BlockTaskRunMixin,
                RequiresMixin,
                RunMixin,
                luigi.Task):
    '''Copy blocks from the inputs to produce the output'''
    
    task_namespace="ariadne_microns_pipeline"
