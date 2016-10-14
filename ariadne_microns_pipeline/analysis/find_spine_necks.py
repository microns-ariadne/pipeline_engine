'''Find spine necks in a ground-truth segmentation'''

import luigi
import numpy as np
from scipy.ndimage import label, grey_erosion, grey_dilation, \
     distance_transform_edt, binary_dilation
from scipy.sparse import coo_matrix

from ..algorithms.morphology import erode_segmentation, SIX_CONNECTED
from ..parameters import VolumeParameter, DatasetLocationParameter
from ..targets.factory import TargetFactory

class FindSpineNecksTask(luigi.Task):
    '''Find the spine necks in a segmentation
    
    This task looks for pieces of a segmentation that are narrower than
    a given diameter. It separates these from the main body of a segmented
    object and returns a segmentation that only contains the necks.
    '''
    task_namespace="ariadne_microns_pipeline"
    
    volume = VolumeParameter(
        description="The volume to be analyzed")
    input_location=DatasetLocationParameter(
        description="the location of the ground-truth segmentation")
    output_location=DatasetLocationParameter(
        description="The location for the spine necks")
    max_diameter_nm = luigi.FloatParameter(
        description="The maximum diameter of a spine neck in nanometers")
    min_volume = luigi.IntParameter(
        default=1000,
        description="The minimum volume for a spine neck")
    xy_nm = luigi.FloatParameter(
        default=4.0,
        description="The size of a voxel in the x/y direction in nanometers")
    z_nm = luigi.FloatParameter(
        default=30.0,
        description="The size of a voxel in the z direction in nanometers")
    
    def input(self):
        yield TargetFactory().get_volume_target(
            volume=self.volume,
            location = self.input_location)
    
    def output(self):
        return TargetFactory().get_volume_target(
            volume=self.volume,
            location = self.output_location)
    
    def run(self):
        input_tgt = self.input().next()
        seg = input_tgt.imread()
        #
        # Erode the segmentation to make sure there is background between
        # every segment
        #
        erode_segmentation(seg, SIX_CONNECTED, in_place=True)
        #
        # The next step is basically an opening operation: an erosion
        # followed by a dilation
        #
        # Find the distance to every foreground voxel
        #
        distance = distance_transform_edt(
            seg > 0,
            sampling=(self.z_nm, self.xy_nm, self.xy_nm))
        erosion = distance > self.max_diameter_nm / 2
        #
        # We use a propagation here - dilate the image, then mask by
        # the eroded segmentation to keep the dilation from spreading
        # across segments.
        #
        # The factor of 2 is heuristic. I originally thought that sqrt(2)
        # would be the right answer...
        #
        n_rounds = int(self.max_diameter_nm / self.xy_nm)
        z = 0 # use the breshnam algorithm to know when to do Z
        opening = erosion
        four_connected = np.array([np.zeros((3, 3), bool),
                                   [[False, True, False],
                                    [True, True, True],
                                    [False, True, False]],
                                   np.zeros((3, 3), bool)])
        for _ in range(n_rounds):
            if z > self.z_nm:
                z -= self.z_nm
                strel = SIX_CONNECTED
            else:
                strel = four_connected
            opening = binary_dilation(opening, structure =strel)
            z += self.xy_nm
        #
        # Goners are the part of the segmentation not in the opening
        #
        goners, gcount = label((seg > 0) & ~ opening)
        #
        # Get rid of small goners
        #
        areas = np.bincount(goners.ravel())
        areas[0] = 0
        is_spine_neck = areas > self.min_volume
        relabeling = np.zeros(len(is_spine_neck), seg.dtype)
        relabeling[is_spine_neck] = np.arange(np.sum(is_spine_neck))+1
        
        spine_necks = relabeling[goners]
        #
        # Output the spine-neck segmentation
        #
        self.output().imwrite(spine_necks)