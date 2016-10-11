'''Match detected neurons to ground-truth'''

import json
import luigi
import numpy as np
from scipy.sparse import coo_matrix

from ..parameters import DatasetLocationParameter, VolumeParameter
from ..targets.factory import TargetFactory
from .utilities import RunMixin, RequiresMixin, SingleThreadedMixin

class MatchNeuronsTaskMixin:
    
    volume = VolumeParameter(
        description="The volume being analyzed")
    gt_location = DatasetLocationParameter(
        description="The location on disk of the ground-truth segmentation")
    detected_location = DatasetLocationParameter(
        description="The location on disk of the result of automatic "
        "segmentation")
    output_location = luigi.Parameter(
        description="The location of the .json file containing the "
                    "correspondences between gt and detected neurons")
    
    def input(self):
        yield TargetFactory().get_volume_target(
            volume=self.volume,
            location=self.gt_location)
        yield TargetFactory().get_volume_target(
            volume=self.volume,
            location=self.detected_location)
    
    def output(self):
        return luigi.LocalTarget(self.output_location)

class MatchNeuronsRunMixin:
    
    def ariadne_run(self):
        '''Record the gt neuron with the maximal overlap to detected'''
        inputs = self.input()
        gt_tgt = inputs.next()
        d_tgt = inputs.next()

        result = {}
        result["volume"] = dict(x=self.volume.x,
                                y=self.volume.y,
                                z=self.volume.z,
                                width=self.volume.width,
                                height=self.volume.height,
                                depth=self.volume.depth)
        #
        # Get flattened versions of the ground-truth and detected space.
        # The pixel-pixel correlations are still maintained and we don't
        # need the spatial info.
        #
        gt_flat = gt_tgt.imread().ravel()
        d_flat = d_tgt.imread().ravel()
        d_max = np.max(d_flat)
        mask = ((gt_flat != 0) & (d_flat != 0))
        gt_flat, d_flat = gt_flat[mask], d_flat[mask]
        if not np.any(mask):
            result["gt"] = [ 0 ] * (d_max+1)
        else:
            #
            # Get # voxels per gt / d pair
            #
            matrix = coo_matrix((np.ones(gt_flat.shape, np.uint32),
                                 (d_flat, gt_flat)))
            matrix.sum_duplicates()
            l_d, l_gt = matrix.nonzero()
            count = matrix.tocsr()[l_d, l_gt].A1
            #
            # Now we find the counts that match the max for their detected label. 
            # Ties go to whomever.
            #
            max_per_d = matrix.max(axis=1).toarray().flatten()
            best = (count == max_per_d[l_d])
            where_best = np.zeros(d_max+1, np.uint32)
            where_best[l_d[best]] = l_gt[best]
            where_best[max_per_d == 0] = 0
            result["gt"] = where_best.tolist()
        with self.output().open("w") as fd:
            json.dump(result, fd)

class MatchNeuronsTask(MatchNeuronsTaskMixin,
                       MatchNeuronsRunMixin,
                       RequiresMixin,
                       RunMixin,
                       SingleThreadedMixin,
                       luigi.Task):
    '''Find the ground-truth neuron that has the most overlap with detected
    
    The output is a .json file with the following keys:
    
    volume: the volume analyzed - a dictionary with keys of x, y, z,
            width, height and depth
    gt: the label of the ground-truth neuron that overlaps or zero if none
        does for each detected neuron.
    '''
    
    task_namespace = "ariadne_microns_pipeline"
        