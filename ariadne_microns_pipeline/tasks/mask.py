import luigi
import numpy as np
from scipy.ndimage import gaussian_filter

from ..targets.factory import TargetFactory
from ..parameters import VolumeParameter, DatasetLocationParameter
from utilities import RequiresMixin, RunMixin


class MaskBorderTaskMixin:
    volume = VolumeParameter(
        description="The volume to mask")
    prob_location = DatasetLocationParameter(
        description="The location of the probability files")
    mask_location = DatasetLocationParameter(
        description="The location of the mask files to be output")
    
    def input(self):
        yield TargetFactory().get_volume_target(
            self.prob_location, self.volume)
    
    def output(self):
        return TargetFactory().get_volume_target(
            self.mask_location, self.volume)
    
    def estimate_memory_usage(self):
        '''Return an estimate of bytes of memory required by this task'''
        v1 = np.prod([1888, 1888, 100]) * 2
        m1 = 716834 * 1000
        v2 = np.prod([1888, 1888, 52]) * 2
        m2 = 468952 * 1000
        #
        # Model is Ax + B where x is the output volume
        #
        B = (v1 * m2 - v2 * m1) / (v1 - v2)
        A = (float(m1) - B) / v1
        v = np.prod([self.volume.width, 
                     self.volume.height, 
                     self.volume.depth])
        return int(A * v + B)


class MaskBorderRunMixin:
    #
    # Optional parameters
    #
    threshold = luigi.IntParameter(
        default = 250,
        description="Mask out voxels whose membrane probability is this value "
                    "or higher (range = 0-256)")
    smoothing_xy = luigi.FloatParameter(
        default=0,
        description="Smoothing in the x and y direction. "
                    "Zero means smoothing off")
    smoothing_z = luigi.FloatParameter(
        default=0,
        description="Smoothing in the Z direction")
    def ariadne_run(self):
        '''Create the mask of pixels to watershed'''
        prob = self.input().next().imread()
        if self.smoothing_xy != 0 or self.smoothing_z != 0:
            prob = gaussian_filter(prob.astype(float32),
                                   sigma=(self.smoothing_z,
                                          self.smoothing_xy,
                                          self.smoothing_xy))
        mask = (prob < self.threshold).astype(np.uint8)
        self.output().imwrite(mask)


class MaskBorderTask(MaskBorderTaskMixin, MaskBorderRunMixin, 
                     RequiresMixin, RunMixin, luigi.Task):

    task_namespace = "ariadne_microns_pipeline"
