'''Stitch a segmentation into a unitary form'''

import enum
import h5py
import json
import luigi
import numpy as np
import os

from ..parameters import VolumeParameter, DatasetLocationParameter
from ..parameters import MultiVolumeParameter
from ..targets.factory import TargetFactory, TFEnums
from .utilities import RunMixin, RequiresMixin, SingleThreadedMixin, to_hashable

class Compression(enum.Enum):
    '''Compression types for HDF5'''
    NONE=0
    GZIP=1
    LZF=2

class StitchSegmentationTaskMixin:
    
    input_volumes=MultiVolumeParameter(
        description="The input segmentation volumes")
    connected_components_location=luigi.Parameter(
        description="The output file from AllConnectedComponentsTask "
            "that gives the local <-> global label correspondences")
    output_volume=VolumeParameter(
        description="The volume of the output")
    output_location=DatasetLocationParameter(
        description="The location of the HDF5 file")
    
    def input(self):
        tf = TargetFactory()
        yield luigi.LocalTarget(self.connected_components_location)
        
        for d in self.input_volumes:
            volume = d["volume"]
            location = d["location"]
            yield tf.get_volume_target(location, volume)
    
    def output(self):
        return TargetFactory().get_volume_target(
            location = self.output_location,
            volume = self.output_volume,
            target_type = TFEnums.use_hdf5_volume)

class StitchSegmentationRunMixin:
    
    xy_chunking = luigi.IntParameter(
        default=2048,
        description="The # of voxels in an HDF5 chunk in the x and y direction")
    z_chunking = luigi.IntParameter(
        default=4,
        description="The # of voxels in an HDF5 chunk in the z direction")
    compression = luigi.EnumParameter(
        enum=Compression,
        default=Compression.GZIP,
        description="The type of compression for the HDF5 file")
    
    def ariadne_run(self):
        output_tgt = self.output()
        x0 = self.output_volume.x
        y0 = self.output_volume.y
        z0 = self.output_volume.z
        x1 = self.output_volume.x1
        y1 = self.output_volume.y1
        z1 = self.output_volume.z1
        chunks=(min(self.z_chunking, self.output_volume.depth),
                min(self.xy_chunking, self.output_volume.height),
                min(self.xy_chunking, self.output_volume.width))
        kwds = dict(chunks=chunks)
        if self.compression == Compression.GZIP:
            kwds["compression"] = "gzip"
        elif self.compression == Compression.LZF:
            kwds["compression"] = "lzf"
        output_tgt.create_volume(dtype=np.uint32, **kwds)
        inputs = self.input()
        
        cc_location = inputs.next()
        cc = json.load(cc_location.open("r"),
                       object_hook = to_hashable)
        #
        # This is a map of volume to local/global labelings
        #
        volume_map = dict(cc["volumes"])
        #
        # Loop over each volume
        #
        for tgt in inputs:
            if tgt.volume.x1 <= x0 or\
               tgt.volume.y1 <= y0 or\
               tgt.volume.z1 <= z0 or\
               tgt.volume.x >= x1 or \
               tgt.volume.y >= y1 or \
               tgt.volume.z >= z1:
                continue
            tmp = np.array(
                volume_map[to_hashable(tgt.volume.to_dictionary())])
            mapping = np.zeros(np.max(tmp[:, 0]) + 1, np.uint32)
            mapping[tmp[:, 0]] = tmp[:, 1]
            seg = mapping[tgt.imread()]
            
            x0a = max(x0, tgt.volume.x)
            x1a = min(x1, tgt.volume.x1)
            y0a = max(y0, tgt.volume.y)
            y1a = min(y1, tgt.volume.y1)
            z0a = max(z0, tgt.volume.z)
            z1a = min(z1, tgt.volume.z1)
            output_tgt.imwrite_part(
                seg[z0a-tgt.volume.z:z1a-tgt.volume.z,
                    y0a-tgt.volume.y:y1a-tgt.volume.y,
                    x0a-tgt.volume.x:x1a-tgt.volume.x],
                x0a, y0a, z0a)
        output_tgt.finish_volume()

class StitchSegmentationTask(StitchSegmentationTaskMixin,
                             StitchSegmentationRunMixin,
                             RunMixin,
                             RequiresMixin,
                             SingleThreadedMixin,
                             luigi.Task):
    '''Stitch a segmentation together from blocks
    
    Create an HDF5 file with the stitched segmentation from many blocks.
    '''
    
    task_namespace = "ariadne_microns_pipeline"