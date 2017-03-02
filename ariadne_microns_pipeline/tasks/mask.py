import luigi
import numpy as np
from scipy.ndimage import gaussian_filter

from .utilities import RequiresMixin, RunMixin, DatasetMixin
from ..targets import DestVolumeReader


class MaskBorderTaskMixin(DatasetMixin):
    prob_loading_plan_path = luigi.Parameter(
        description="The location of the probability files")
    
    def input(self):
        for tgt in DestVolumeReader(self.prob_loading_plan_path) \
            .get_source_targets():
            yield tgt
    
    def estimate_memory_usage(self):
        '''Return an estimate of bytes of memory required by this task'''
        v1 = np.prod([1888, 1416, 70])
        m1 = 716834 * 1000
        v2 = np.prod([1888, 1416, 42])
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
        prob = DestVolumeReader(self.prob_loading_plan_path).imread()
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
