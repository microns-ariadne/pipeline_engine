'''Download a volume from butterfly into HDF5'''

import luigi
import numpy as np
from .utilities import RunMixin
from ..targets.butterfly_target \
     import ButterflyChannelTarget, get_butterfly_plane_from_channel
from ..targets.factory import TargetFactory
from ..parameters import VolumeParameter, DatasetLocationParameter

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
    volume=VolumeParameter(
        description="The volume to download")
    destination=DatasetLocationParameter(
        description="The destination for the dataset.")
    url = luigi.Parameter(
        default="http://localhost:2001/api",
        description="URL of the REST endpoint of the Butterfly server")
    resolution = luigi.IntParameter(
        default=0,
        description="The MIPMAP resolution of the image to be retrieved.")
    
    def output(self):
        return TargetFactory().get_volume_target(
            self.destination, self.volume)

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

class DownloadFromButterflyRunMixin:
    '''Perform the aggregation of butterfly planes into a volume'''
    
    def ariadne_run(self):
        '''Copy data from the Butterfly planes to the HDF5 target'''
        
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
