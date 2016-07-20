import luigi
import numpy as np
from scipy.ndimage import distance_transform_edt

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


class MaskBorderRunMixin:
    #
    # Optional parameters
    #
    border_width = luigi.IntParameter(
        default=30,
        description="Border to potentially exclude from watershed")
    close_width = luigi.IntParameter(
        default=5,
        description="Radius of the close() operation")
    
    def ariadne_run(self):
        '''Create the mask of pixels to watershed'''
        #
        # TO_DO: implement this when we understand what it is
        #
        prob = self.input().next().imread()
        mask = np.ones(prob.shape, np.uint8)
        self.output().imwrite(mask)


class MaskBorderTask(MaskBorderTaskMixin, MaskBorderRunMixin, 
                     RequiresMixin, RunMixin, luigi.Task):

    task_namespace = "ariadne_microns_pipeline"
    
