'''Match detected synapses against ground truth'''

import enum
import json
import luigi
import numpy as np

from ..algorithms.evaluation import match_synapses_by_distance
from ..algorithms.evaluation import match_synapses_by_overlap
from ..parameters import VolumeParameter, DatasetLocationParameter,\
     EMPTY_DATASET_LOCATION, is_empty_dataset_location
from ..targets.factory import TargetFactory
from .utilities import RunMixin, RequiresMixin, SingleThreadedMixin

class MatchMethod(enum.Enum):
    
    '''Match synapses that overlap the most'''
    overlap = 1
    
    '''Match synapses whose centroids are the closest'''
    distance = 2

class MatchSynapsesTaskMixin:
    
    volume = VolumeParameter(
        description="Volume containing gt and detected synapses to be matched")
    gt_location = DatasetLocationParameter(
        description="Location of the ground truth on disk")
    detected_location = DatasetLocationParameter(
        description="Location of the synapse detections on disk")
    mask_location = DatasetLocationParameter(
        default=EMPTY_DATASET_LOCATION,
        description="Location of the mask of the annotated volume")
    output_location = luigi.Parameter(
        description=
        "Location for the .json file containing the synapse matches")
    
    def has_mask(self):
        return not is_empty_dataset_location(self.mask_location)
    
    def input(self):
        yield TargetFactory().get_volume_target(
            location=self.gt_location,
            volume=self.volume)
        yield TargetFactory().get_volume_target(
            location=self.detected_location,
            volume=self.volume)
        if self.has_mask():
            yield TargetFactory().get_volume_target(
                location = self.mask_location,
                volume=self.volume)
    
    def output(self):
        return luigi.LocalTarget(self.output_location)

class MatchSynapsesRunMixin:
    
    match_method = luigi.EnumParameter(
        default=MatchMethod.overlap,
        enum=MatchMethod,
        description="The method used for matching detected synapses to "
        "ground truth ones")
    xy_nm = luigi.FloatParameter(
        default=4.0,
        description=
        "The size of a voxel in nanometers in the x and y directions")
    z_nm = luigi.FloatParameter(
        default=30.0,
        description="The size of a voxel in nanometers in the z direction")
    max_distance = luigi.FloatParameter(
        default=100.,
        description="The maximum allowed distance between matching centroids")
    min_overlap_pct = luigi.FloatParameter(
        default=25.,
        description="The percentage of the total object that must overlap "
        "to be considered a match.")
    
    def ariadne_run(self):
        inputs = self.input()
        gt_tgt = inputs.next()
        d_tgt = inputs.next()
        gt = gt_tgt.imread()
        d = d_tgt.imread()
        if self.has_mask():
            #
            # Erase any synapse outside of the annotated volume
            #
            mask_tgt = inputs.next()
            mask = mask_tgt.imread()
            d[mask == 0] = 0
        d_hist = np.bincount(d.flatten())
        d_hist[0] = 0
        d_labels = np.where(d_hist)[0]
            
        if self.match_method == MatchMethod.overlap:
            matching_d, matching_gt = match_synapses_by_overlap(
                gt, d, self.min_overlap_pct)
        else:
            matching_d, matching_gt = match_synapses_by_distance(
                gt, d, self.xy_nm, self.z_nm, self.max_distance)
        volume = dict(x=self.volume.x,
                      y=self.volume.y,
                      z=self.volume.z,
                      width=self.volume.width,
                      height=self.volume.height,
                      depth = self.volume.depth)
        with self.output().open("w") as fd:
            json.dump(dict(
                volume=volume,
                detected_labels=d_labels.tolist(),
                detected_per_gt=matching_d.tolist(),
                gt_per_detected=matching_gt.tolist()), fd)

class MatchSynapsesTask(MatchSynapsesTaskMixin,
                        MatchSynapsesRunMixin,
                        RunMixin,
                        RequiresMixin,
                        SingleThreadedMixin,
                        luigi.Task):
    '''Match ground-truth and detected synapses
    
    This task takes a volume annotated with ground-truth synapses
    and those detected automatically. It produces a JSON file with the
    following keys:
    
    volume: the volume analyzed. This is a dictionary of the same format
            as the serialization of the VolumeParameter (x, y, z, width,
            height and depth).
    detected_per_gt: a sequence of matching detected synapses for each
                     ground-truth synapse. "detected_per_gt[idx]" is the
                     label number in the detected volume of the detected
                     synapse that best matches the ground-truth synapse
                     with label number "idx". If "detected_per_gt[idx]" is
                     zero, then the ground-truth synapse with label number,
                     "idx", has no matching detected synapse.
    gt_per_detected: a sequence of matching ground-truth synapses for each
                     detected. See "detected_per_gt" for explanation.
    detected_labels: the labels taken into consideration when matching.
    '''
    
    task_namespace = "ariadne_microns_pipeline"
