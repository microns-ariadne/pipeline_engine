'''Find synapse objects given a probability map of them'''

import luigi
import numpy as np
import rh_logger

from .utilities import RequiresMixin, RunMixin, SingleThreadedMixin, \
     DatasetMixin
from ..algorithms.segmentation import segment_vesicle_style
from ..algorithms.morphology import erode_segmentation
from ..parameters import EMPTY_LOCATION
from ..targets import DestVolumeReader

class FindSynapsesTaskMixin:
    
    wants_dual_probability_maps = luigi.BoolParameter(
        description="Set this if both a transmitter and receptor probability "
                    "map are provided. Otherwise a single synapse probability "
                    "map is used.")
    synapse_map_loading_plan_path = luigi.Parameter(
        default = EMPTY_LOCATION,
        description="The location of the synapse probability map")
    transmitter_map_loading_plan_path = luigi.Parameter(
        default = EMPTY_LOCATION,
        description = "The location of the synapse transmitter probability map")
    receptor_map_loading_plan = luigi.Parameter(
        default = EMPTY_LOCATION,
        description = "The location of the synapse receptor probability map")
    neuron_segmentation_loading_plan = luigi.Parameter(
        default=EMPTY_LOCATION,
        description="The location of the segmented neurons.")
    
    def input(self):
        if self.wants_dual_probability_maps:
            loading_plans = [self.transmitter_map_loading_plan_path,
                             self.receptor_map_loading_plan]
        else:
            loading_plans = [self.synapse_map_loading_plan_path]
        if not is_empty_dataset_location(self.neuron_segmentation_loading_plan):
            loading_plans.append(self.neuron_segmentation_loading_plan)
        for loading_plan in loading_plans:
            for tgt in DestVolumeReader(loading_plan).get_source_targets():
                yield tgt
    
    def estimate_memory_usage(self):
        '''Return an estimate of bytes of memory required by this task'''
        v1 = np.prod([1888, 1416, 70])
        m1 = 3722953 * 1000
        v2 = np.prod([1416, 1888, 42])
        m2 = 2050723 * 1000
        #
        # Model is Ax + B where x is the output volume
        #
        B = (v1 * m2 - v2 * m1) / (v1 - v2)
        A = (float(m1) - B) / v1
        v = np.prod([self.volume.width, 
                     self.volume.height, 
                     self.volume.depth])
        return int(A * v + B)
    
    
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
    erode_with_neurons = luigi.BoolParameter(
        description="Exclude areas within neurons")
    
    def ariadne_run(self):
        if self.wants_dual_probability_maps:
            # Take the sum of the transmitter and receptor probabilities
            volume = \
                DestVolumeReader(
                    self.transmitter_map_loading_plan_path).imread() + \
                DestVolumeReader(
                    self.receptor_map_loading_plan_path).imread()
        else:
            volume = DestVolumeReader(self.synapse_map_loading_plan_path) \
                .imread()
        if self.erode_with_neurons:
            #
            # Exclude the innards of the neuron from consideration
            #
            neuron_segmentation = DestVolumeReader(
                self.neuron_segmentation_loading_plan_path).imread()
            strel = np.ones((self.erosion_z * 2 + 1,
                             self.erosion_xy * 2 + 1,
                             self.erosion_xy * 2 + 1), bool)
            erode_segmentation(neuron_segmentation, strel, in_place=True)
            volume[neuron_segmentation != 0] = 0
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
    