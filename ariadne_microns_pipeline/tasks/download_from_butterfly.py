'''Download a volume from butterfly into HDF5'''

import luigi
import numpy as np
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
    
    def output(self):
        return TargetFactory().get_volume_target(
            self.destination, self.volume)


class DownloadFromButterflyRunMixin:
    '''Perform the aggregation of butterfly planes into a volume'''
    
    def ariadne_run(self):
        '''Copy data from the Butterfly planes to the HDF5 target'''
        
        volume = np.zeros(
            (self.volume.depth, self.volume.height, self.volume.width), 
            np.uint8)
        inputs = self.input()
        channel_target = ButterflyChannelTarget(
            self.experiment, self.sample, self.dataset, self.channel, self.url)
        plane_targets = inputs[1:]
        for z in range(self.volume.z, self.volume.z1):
            plane = get_butterfly_plane_from_channel(
                channel_target, self.volume.x, self.volume.y, z,
                self.volume.width, self.volume.height)
            volume[z - self.volume.z] = plane.imread()
        
        self.output().imwrite(volume)


class DownloadFromButterflyTask(DownloadFromButterflyRunMixin,
                                DownloadFromButterflyTaskMixin,
                                luigi.ExternalTask):
    '''A task for downloading butterfly planes into a volume'''
    
    task_namespace = "ariadne_microns_pipeline"
    
    def run(self):
        self.ariadne_run()
