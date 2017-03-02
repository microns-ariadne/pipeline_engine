'''Match detected synapses against ground truth'''

import enum
import json
import luigi
import numpy as np

from ..algorithms.evaluation import match_synapses_by_distance
from ..algorithms.evaluation import match_synapses_by_overlap
from ..parameters import EMPTY_LOCATION
from ..targets import DestVolumeReader
from .utilities import RunMixin, RequiresMixin, SingleThreadedMixin

class MatchMethod(enum.Enum):
    
    '''Match synapses that overlap the most'''
    overlap = 1
    
    '''Match synapses whose centroids are the closest'''
    distance = 2

class MatchSynapsesTaskMixin:
    
    gt_loading_plan_path = luigi.Parameter(
        description="Location of the ground truth on disk")
    detected_loading_plan_path = luigi.Parameter(
        description="Location of the synapse detections on disk")
    mask_loading_plan_path = luigi.Parameter(
        default=EMPTY_LOCATION,
        description="Location of the mask of the annotated volume")
    output_location = luigi.Parameter(
        description=
        "Location for the .json file containing the synapse matches")
    
    def has_mask(self):
        return self.mask_location != EMPTY_LOCATION
    
    def input(self):
        loading_plans = [self.gt_loading_plan_path, 
                         self.detected_loading_plan_path]
        if self.has_mask:
            loading_plans.append(self.mask_loading_plan_path)
        for loading_plan in loading_plans:
            for tgt in DestVolumeReader(loading_plan).get_source_targets():
                yield tgt
    
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
        gt_tgt = DestVolumeReader(self.gt_loading_plan_path)
        d_tgt = DestVolumeReader(self.detected_loading_plan_path)
        gt = gt_tgt.imread()
        d = d_tgt.imread()
        if self.has_mask():
            #
            # Erase any synapse outside of the annotated volume
            #
            mask_tgt = DestVolumeReader(self.mask_loading_plan_path)
            mask = mask_tgt.imread()
            d[mask == 0] = 0
            gt[mask == 0] = 0
        d_hist = np.bincount(d.flatten())
        d_hist[0] = 0
        d_labels = np.where(d_hist > 0)[0]
        
        gt_hist = np.bincount(gt.flatten())
        gt_hist[0] = 0
        gt_labels = np.where(gt_hist > 0)[0]
            
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
                gt_labels=gt_labels.tolist(),
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
