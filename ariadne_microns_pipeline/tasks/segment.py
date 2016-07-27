import luigi
import numpy as np
from scipy.ndimage import gaussian_filter, label

from ..algorithms import watershed
from ..targets.factory import TargetFactory
from ..parameters import VolumeParameter, DatasetLocationParameter
from utilities import RequiresMixin, RunMixin, SingleThreadedMixin

class SegmentTaskMixin:
    '''Segment a 3d volume
    
    Given a probability map, find markers and perform a segmentation on
    that volume.
    '''
    volume = VolumeParameter(
        description="The volume to segment")
    prob_location = DatasetLocationParameter(
        description="The location of the probability volume")
    seed_location = DatasetLocationParameter(
        description="The location of the seeds for the watershed")
    mask_location = DatasetLocationParameter(
        description="The location of the mask volume")
    output_location = DatasetLocationParameter(
        description="The location for the output segmentation")

    def input(self):
        yield TargetFactory().get_volume_target(
            location=self.prob_location, volume=self.volume)
        yield TargetFactory().get_volume_target(
            location=self.seed_location, volume=self.volume)
        yield TargetFactory().get_volume_target(
            location=self.mask_location,
            volume=self.volume)
    def output(self):
        return TargetFactory().get_volume_target(
            location=self.output_location, volume=self.volume)

class SegmentRunMixin:
    
    sigma_xy = luigi.FloatParameter(
        description=
        "The sigma of the smoothing Gaussian in the x & y directions",
        default=3)
    sigma_z = luigi.FloatParameter(
        description=
        "The sigma of the smoothing Gaussian in the z direction",
        default=.4)

    def ariadne_run(self):
        prob_volume, seed_volume, mask_volume = list(self.input())
        seg_volume = self.output()
        prob = prob_volume.imread()
        labels = seed_volume.imread()
        mask = mask_volume.imread() != 0
        labels[~ mask] = 0
        prob[~mask] = 255
        smoothed = gaussian_filter(
            prob, (self.sigma_z, self.sigma_xy, self.sigma_xy))
        seg = watershed(smoothed, labels)
        seg_volume.imwrite(seg.astype(np.uint16))

class SegmentTask(SegmentTaskMixin, SegmentRunMixin, RequiresMixin, 
                  RunMixin, SingleThreadedMixin, luigi.Task):
    
    task_namespace = "ariadne_microns_pipeline"
