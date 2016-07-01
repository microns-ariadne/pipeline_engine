'''Download a volume from butterfly into HDF5'''

import luigi
import numpy as np
from ..targets.butterfly_target \
     import ButterflyChannelTarget, get_butterfly_plane_from_channel
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
    x = luigi.IntParameter(
        description="The x-offset of the sub-volume within the dataset.")
    y = luigi.IntParameter(
        description="The y-offset of the sub-volume within the dataset.")
    z = luigi.IntParameter(
        description="The z-offset of the sub-volume within the dataset.")
    width = luigi.IntParameter(
        description="The width of the subvolume.")
    height = luigi.IntParameter(
        description="The height of the subvolume.")
    depth = luigi.IntParameter(
        description="The # of planes in the subvolume.")
    dest_paths = luigi.ListParameter(
        description="The names of the root tempfile directories")
    dest_dataset = luigi.Parameter(
        description="The name of the dataset within the HDF file")
    dest_pattern = luigi.Parameter(
        description="A pattern for str.format(). The variables available "
        'are "x", "y" and "z". Example: "{x:04}_{y:04}_{z:04}" yields '
        '"0001_0002_0003.h5" for a plane with X offset 1, Y offset 2 and '
        'Z offset 3',
        default="{x:04}_{y:04}_{z:04}")
    url = luigi.Parameter(
        default="http://localhost:2001/api",
        description="URL of the REST endpoint of the Butterfly server")
    
    def output(self):
        return TargetFactory().get_volume_target(
            self.dest_paths, self.dest_dataset, self.dest_pattern,
            self.x, self.y, self.z, self.width, self.height, self.depth)


class DownloadFromButterflyRunMixin:
    '''Perform the aggregation of butterfly planes into a volume'''
    
    def ariadne_run(self):
        '''Copy data from the Butterfly planes to the HDF5 target'''
        
        volume = np.zeros((self.depth, self.height, self.width), np.uint8)
        inputs = self.input()
        channel_target = ButterflyChannelTarget(
            self.experiment, self.sample, self.dataset, self.channel, self.url)
        plane_targets = inputs[1:]
        for z in range(self.z, self.z+self.depth):
            plane = get_butterfly_plane_from_channel(
                channel_target, self.x, self.y, z,
                self.width, self.height)
            volume[z - self.z] = plane.imread()
        
        self.output().imwrite(volume)


class DownloadFromButterflyTask(DownloadFromButterflyRunMixin,
                                DownloadFromButterflyTaskMixin,
                                luigi.ExternalTask):
    '''A task for downloading butterfly planes into a volume'''
    
    task_namespace = "ariadne_microns_pipeline"
    
    def run(self):
        self.ariadne_run()
