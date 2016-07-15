'''Tasks for finding connected components across blocks'''

import json
import luigi
import numpy as np

from .utilities import RequiresMixin
from ..parameters import VolumeParameter, DatasetLocationParameter
from ..targets.factory import TargetFactory

class ConnectedComponentsTaskMixin:
    
    volume1 = VolumeParameter(
        description="The volume for the first of the two segmentations")
    location1 = DatasetLocationParameter(
        description="The location of the first of the two segmentations")
    volume2 = VolumeParameter(
        description="The volume for the second of the two segmentations")
    location2 = DatasetLocationParameter(
        description="The location of the second of the two segmentations")
    overlap_volume = VolumeParameter(
        description="Look at the concordance between segmentations "
        "in this volume")
    output_location = luigi.Parameter(
        description="The location for the JSON file containing the concorances")

    def input(self):
        tf = TargetFactory()
        yield tf.get_volume_target(self.location1, self.volume1)
        yield tf.get_volume_target(self.location2, self.volume2)
    
    def output(self):
        return luigi.LocalTarget(self.output_location)
    
class ConnectedComponentsRunMixin:
    
    def ariadne_run(self):
        '''Look within the overlap volume to find the concordances
        
        We load the two volumes and note the unique pairs of labels
        that appear together at the same voxel for the voxels in the
        overlap volume.
        '''
        volume1, volume2 = list(self.input())
        cutout1 = volume1.imread_part(self.overlap_volume.x,
                                      self.overlap_volume.y,
                                      self.overlap_volume.z,
                                      self.overlap_volume.width,
                                      self.overlap_volume.height,
                                      self.overlap_volume.depth)
        
        cutout2 = volume2.imread_part(self.overlap_volume.x,
                                      self.overlap_volume.y,
                                      self.overlap_volume.z,
                                      self.overlap_volume.width,
                                      self.overlap_volume.height,
                                      self.overlap_volume.depth)
        #
        # Order the two cutouts by first segmentation, then second.
        # Sort them using the order and then take only indices i where
        # cutouts[i] != cutouts[i+1]
        #
        cutouts = np.column_stack((cutout1.ravel(), cutout2.ravel()))
        order = np.lexsort((cutouts[:, 1], cutouts[:, 0]))
        cutouts = cutouts[order]
        first = np.where(np.any(cutouts[:-1, :] != cutouts[1:, :], 1))
        unique = cutouts[first]
        as_list = [ (a, b) for a, b in unique.tolist()]
        with self.output().open("w") as fd:
            json.dump(as_list, fd)

class ConnectedComponentsTask(ConnectedComponentsTaskMixin,
                              ConnectedComponentsRunMixin,
                              RequiresMixin,
                              luigi.Task):
    '''This task finds the connections between the segmentations of two volumes
    
    Given segmentation #1 and segmentation #2 and an overlapping volume
    look at the labels in segmentation #1 and #2 at each pixel. These are
    the labels that are connected between the volumes. This task finds the
    unique labels between the segmentations and stores them in a JSON file.
    
    The connected components of the whole volume can be found by loading
    all of the JSON files and assigning each block's labels to a global
    label that may be shared between segments.
    '''
    
    task_namespace = 'ariadne_microns_pipeline'
    
    def run(self):
        self.ariadne_run()