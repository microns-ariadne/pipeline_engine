'''Download a volume from butterfly into HDF5'''

import cv2
import luigi
import json
import numpy as np
import rh_logger
from .utilities import RunMixin, DatasetMixin
from ..parameters import VolumeParameter
from ..targets.butterfly_target \
     import ButterflyChannelTarget, get_butterfly_plane_from_channel
from ..targets import SrcVolumeTarget
from ..targets.factory import TargetFactory

class DownloadFromButterflyTaskMixin:
    '''Download a volume from Butterfly'''
    
    experiment = luigi.Parameter(
        description="The Butterfly experiment that produced the dataset")
    sample = luigi.Parameter(
        description="The ID of the biological sample that was imaged")
    dataset = luigi.Parameter(
        description="The name of the volume that was imaged")
    channel = luigi.Parameter(
        description="The name of the channel from which we take data")
    url = luigi.Parameter(
        default="http://localhost:2001/api",
        description="URL of the REST endpoint of the Butterfly server")
    resolution = luigi.IntParameter(
        default=0,
        description="The MIPMAP resolution of the image to be retrieved.")
    volume = VolumeParameter(description="The voxel location to download")
    
    def estimate_memory_usage(self):
        '''Return an estimate of bytes of memory required by this task'''
        v1 = np.prod([1928, 1928, 102]) / 2**(self.resolution*2)
        m1 = 510271 * 1000
        v2 = np.prod([1928, 1928, 54])
        m2 = 346971 * 1000
        #
        # Model is Ax + B where x is volume in voxels
        #
        B = (v1 * m2 - v2 * m1) / (v1 - v2)
        A = (float(m1) - B) / v1
        v = np.prod([self.volume.width, self.volume.height, self.volume.depth])
        #
        # Make a guess as to the bit depth of the output based on the channel
        #
        if self.channel != "raw":
            v = v * 4
        return int(A * v + B)

class DownloadFromButterflyRunMixin(DatasetMixin):
    '''Perform the aggregation of butterfly planes into a volume'''
    
    def ariadne_run(self):
        '''Copy data from the Butterfly planes to the target'''
        
        haz_volume = False
        inputs = self.input()
        channel_target = ButterflyChannelTarget(
            self.experiment, self.sample, self.dataset, self.channel, self.url)
        plane_targets = inputs[1:]
        for z in range(self.volume.z, self.volume.z1):
                
            plane = get_butterfly_plane_from_channel(
                channel_target, self.volume.x, self.volume.y, z,
                self.volume.width, self.volume.height, self.resolution)
            img = plane.imread()
            if not haz_volume:
                volume = np.zeros(
                    (self.volume.depth, self.volume.height, self.volume.width), 
                    img.dtype)
                haz_volume = True
            volume[z - self.volume.z] = img
        
        self.output().imwrite(volume)

class DownloadFromButterflyTask(DownloadFromButterflyRunMixin,
                                DownloadFromButterflyTaskMixin,
                                RunMixin,
                                luigi.ExternalTask):
    '''A task for downloading butterfly planes into a volume'''
    
    task_namespace = "ariadne_microns_pipeline"
    
    def process_resources(self):
        '''Report Butterfly's resource requirements
        
        Butterfly needs one "butterfly" resource
        '''
        resources = self.resources.copy()
        resources["butterfly"] = 1
        return resources

######################################################
#
# Local Butterfly is a standard that bypasses the Butterfly
# server, replacing it with an index file to the Z sections
# and their directories.
#
# The index file is a json dictionary with the following keys
#
# * sections: a list where each list element is a string, suitable for 
#   formatting using s.format(row=xidx, column=yidx)
# * dimensions: a dictionary with the following keys
#   * width - the width of a tile
#   * height - the height of a tile
#   * n_rows - # of tiles in the x direction
#   * n_columns - # of tiles in the y direction
#   * dtype - data type suitable for numpy, e.g. "uint8"
#
#########################################################

class LocalButterflyTaskMixin(DatasetMixin):
    index_file = luigi.Parameter(
        description="The path to the index file for the dataset")

class LocalButterflyRunMixin:
    
    def ariadne_run(self):
        '''Load a volume and write to the storage plan'''
        tgt = self.output()
        assert isinstance(tgt, SrcVolumeTarget)
        volume = tgt.volume
        index = json.load(open(self.index_file))
        dimensions = index["dimensions"]
        dtype = getattr(np, dimensions["dtype"])
        result = np.zeros((volume.depth, volume.height, volume.width), dtype)
        n_rows = dimensions["n_rows"]
        n_columns = dimensions["n_columns"]
        width = dimensions["width"]
        height = dimensions["height"]
        column0 = volume.x / width
        column1 = int(np.ceil(float(volume.x1) / width))
        row0 = volume.y / height
        row1 = int(np.ceil(float(volume.y1) / height))
        for z in range(volume.z, volume.z1):
            pattern = index["sections"][z]
            for row in range(row0, row1):
                y0src = row * height
                y1src = y0src + height
                for column in range(column0, column1):
                    x0src = column * width
                    x1src = x0src + width
                    
                    path = pattern.format(row=row+1, column=column+1)
                    tile = cv2.imread(
                        path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                    if x0src >= volume.x:
                        x0s = 0
                        x0d = x0src - volume.x
                    else:
                        x0s = volume.x - x0src
                        x0d = 0
                    if x1src <= volume.x1:
                        x1s = width
                        x1d = x1src - volume.x
                    else:
                        x1s = volume.x1 - x0src
                        x1d = volume.width
                    if y0src >= volume.y:
                        y0s = 0
                        y0d = y0src - volume.y
                    else:
                        y0s = volume.y - y0src
                        y0d = 0
                    if y1src <= volume.y1:
                        y1s = height
                        y1d = y1src - volume.y
                    else:
                        y1s = volume.y1 - y0src
                        y1d = volume.height
                    if tile is not None:
                        result[z-volume.z, y0d:y1d, x0d:x1d] = \
                            tile[y0s:y1s, x0s:x1s]
                    else:
                        result[z-volume.z, y0d:y1d, x0d:x1d] = 255
                        rh_logger.logger.report_event(
                            "Skipping tile: %s" % path)
        tgt.imwrite(result)

class LocalButterflyTask(LocalButterflyRunMixin,
                         LocalButterflyTaskMixin,
                         RunMixin,
                         luigi.ExternalTask):
    '''Get source data via local process
    
    Local Butterfly is a standard that bypasses the Butterfly 
    server, replacing it with an index file to the Z sections
    and their directories.
    The index file is a json dictionary with the following keys

    * sections: a list where each list element is a string, suitable for 
    formatting using s.format(row=xidx, column=yidx)
    
    * dimensions: a dictionary with the following keys
    
       * width - the width of a tile
       
       * height - the height of a tile
       
       * n_rows - # of tiles in the x direction
          
       * n_columns - # of tiles in the y direction
       
       * dtype - data type suitable for numpy, e.g. "uint8"
'''

    task_namespace = "ariadne_microns_pipeline"

    def process_resources(self):
        '''Report Butterfly's resource requirements

        Butterfly needs one "butterfly" resource
        '''
        resources = self.resources.copy()
        resources["butterfly"] = 1
        return resources
