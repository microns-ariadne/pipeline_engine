'''Filter a segmentation by area

'''

import luigi
import numpy as np
import rh_logger

from .utilities import RequiresMixin, RunMixin, SingleThreadedMixin, \
     DatasetMixin
from ..targets import DestVolumeReader

class FilterSegmentationTaskMixin:
    input_loading_plan_path = luigi.Parameter(
        description = "The location of the input segmentation")
    
    def input(self):
        for tgt in DestVolumeReader(self.input_loading_plan_path) \
            .get_source_targets():
            yield tgt
    
class FilterSegmentationRunMixin(DatasetMixin):
    min_area = luigi.IntParameter(
        description="The minimum area for a segment")
    
    def ariadne_run(self):
        segmentation = DestVolumeReader(self.input_loading_plan_path).imread()
        #
        # Get the # of voxels per segment
        #
        counts = np.bincount(segmentation.flatten())
        #
        # Hack: make sure that "0" (no segment) fails the test
        # Then perform the test to get the indices of the good segments
        #
        counts[0] = -1
        good_counts = np.where(counts > self.min_area)[0]
        #
        # Create a mapping from old segmentation to new with only the
        # big-enough segments having #s > 0
        #
        mapping = np.zeros(counts.shape[0], segmentation.dtype)
        mapping[good_counts] = np.arange(len(good_counts)) + 1
        #
        # Perform the mapping
        segmentation = mapping[segmentation]
        self.output().imwrite(segmentation)

class FilterSegmentationTask(FilterSegmentationTaskMixin,
                             FilterSegmentationRunMixin,
                             RequiresMixin,
                             RunMixin,
                             SingleThreadedMixin,
                             luigi.Task):
    '''Filter out segments with small areas'''
    
    task_namespace = "ariadne_microns_pipeline"