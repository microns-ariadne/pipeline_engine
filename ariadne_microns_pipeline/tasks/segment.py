import luigi
import numpy as np
from scipy.ndimage import gaussian_filter, label

from .find_seeds import Dimensionality
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
    dimensionality = luigi.EnumParameter(
        enum=Dimensionality,
        description="Do either a 2D or 3D watershed, depending on this param")

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
        if self.dimensionality == Dimensionality.D3:
            seg = watershed(smoothed, labels)
        else:
            seg = np.zeros(smoothed.shape, np.uint16)
            for z in range(smoothed.shape[0]):
                seg[z:z+1] = watershed(smoothed[z:z+1], labels[z:z+1])
        seg_volume.imwrite(seg.astype(np.uint16))

class SegmentTask(SegmentTaskMixin, SegmentRunMixin, RequiresMixin, 
                  RunMixin, SingleThreadedMixin, luigi.Task):
    
    task_namespace = "ariadne_microns_pipeline"

class SegmentCCTaskMixin:
    
    volume = VolumeParameter(
        description="The volume to be segmented")
    prob_location = DatasetLocationParameter(
        description="The location of the probability volume")
    mask_location = DatasetLocationParameter(
        description="The location of the mask volume")
    output_location = DatasetLocationParameter(
        description="The location for the output segmentation")
    
    def input(self):
        yield TargetFactory().get_volume_target(
            location = self.prob_location,
            volume = self.volume)
        yield TargetFactory().get_volume_target(
            location = self.mask_location,
            volume = self.volume)
    
    def output(self):
        return TargetFactory().get_volume_target(
            location = self.output_location,
            volume = self.volume)

class SegmentCC2DRunMixin:    

    threshold = luigi.IntParameter(
        default=190,
        description="The probability threshold (from 0-255) to use as a"
        "cutoff for (not) membrane")

    def ariadne_run(self):
        prob_target, mask_target = list(self.input())
        threshold = self.threshold
        fg =  mask_target.imread() & (prob_target.imread() < threshold)
        labels = np.zeros(fg.shape, np.uint16)
        offset = 0
        for i in range(labels.shape[0]):
            l, count = label(fg[i])
            m = l != 0
            labels[i, m] = l[m] + offset
            offset += count
        self.output().imwrite(labels)

class SegmentCC2DTask(SegmentCCTaskMixin,
                    SegmentCC2DRunMixin,
                    RequiresMixin,
                    RunMixin,
                    SingleThreadedMixin,
                    luigi.Task):
    '''The Segment2DTask performs 2D connected components on membrane probs
    
    The task breaks a volume into X/Y planes. It thresholds the membrane
    probabilities and then it finds the connected components in the plane,
    using non-membrane as the foreground.
    '''