'''Find synapse objects given a probability map of them'''

import luigi
import numpy as np
import rh_logger

from .utilities import RequiresMixin, RunMixin, SingleThreadedMixin
from ..algorithms.segmentation import segment_vesicle_style
from ..algorithms.morphology import erode_segmentation
from ..parameters import VolumeParameter, DatasetLocationParameter
from ..targets.factory import TargetFactory

class FindSynapsesTaskMixin:
    
    volume = VolumeParameter(
        description="The volume to be segmented")
    input_location = DatasetLocationParameter(
        description="The location of the probability map")
    neuron_segmentation = DatasetLocationParameter(
        description="The location of the segmented neurons.")
    output_location = DatasetLocationParameter(
        description="The location for the segmentation")
    
    def input(self):
        yield TargetFactory().get_volume_target(
            location=self.input_location,
            volume=self.volume)
        yield TargetFactory().get_volume_target(
            location=self.neuron_segmentation,
            volume = self.volume)
    
    def output(self):
        return TargetFactory().get_volume_target(
            location = self.output_location,
            volume = self.volume)
    
class FindSynapsesRunMixin:
    
    erosion_xy = luigi.IntParameter(
        default=4,
        description = "# of pixels to erode the neuron segmentation in the "
                      "X and Y direction.")
    erosion_z = luigi.IntParameter(
        default=1,
        description = "# of pixels to erode the neuron segmentation in the "
                      "Z direction.")
    sigma_xy = luigi.FloatParameter(
        default=4,
        description="The sigma of the smoothing Gaussian in the X and Y"
        "directions")
    sigma_z = luigi.FloatParameter(
        default = .5,
        description="The sigma of the smoothing Gaussian in the Z direction")
    threshold = luigi.IntParameter(
        default = 127,
        description = "The threshold for classifying a voxel as synapse")
    min_size_2d = luigi.IntParameter(
        default=25,
        description="Remove isolated synapse foreground in a plane if "
        "less than this # of pixels")
    max_size_2d = luigi.IntParameter(
        default=10000,
        description = "Remove large patches of mislabeled synapse in a plane "
        "that have an area greater than this")
    min_size_3d = luigi.IntParameter(
        default=500,
        description = "Minimum size in voxels of a synapse")
    min_slice = luigi.IntParameter(
        default=3,
        description="Minimum acceptable size of a synapse in the Z direction")
    
    def ariadne_run(self):
        volume, neuron_segmentation = \
            [ _.imread() for _ in self.input()]
        #
        # Exclude the innards of the neuron from consideration
        #
        strel = np.ones((self.erosion_z * 2 + 1,
                         self.erosion_xy * 2 + 1,
                         self.erosion_xy * 2 + 1), bool)
        erode_segmentation(neuron_segmentation, strel, in_place=True)
        volume[neuron_segmentation == 0] = 0
        #
        # Perform the segmentation on the synapse probability map
        #
        segmentation = segment_vesicle_style(
            volume,
            self.sigma_xy,
            self.sigma_z,
            self.threshold,
            self.min_size_2d,
            self.max_size_2d,
            self.min_size_3d,
            self.min_slice)
        self.output().imwrite(segmentation)

class FindSynapsesTask(FindSynapsesTaskMixin,
                       FindSynapsesRunMixin,
                       RequiresMixin,
                       RunMixin,
                       SingleThreadedMixin,
                       luigi.Task):
    '''Find synapse objects, given a synapse probability map
    
    The neuron segmentation is eroded and the synapse probability map is
    masked with the inverse of it in order to restrict the synapses to the
    membrane surrounding the neurons. The synapses are then found using
    a heuristic that discards isolated synapse voxels.
    '''
    
    task_namespace = 'ariadne_microns_pipeline'
    