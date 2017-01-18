import enum
import luigi
import numpy as np
import rh_logger

from ..algorithms.morphology import parallel_distance_transform
from ..algorithms.morphology import erode_segmentation
from ..parameters import VolumeParameter, DatasetLocationParameter
from ..targets.factory import TargetFactory
from .utilities import RequiresMixin, RunMixin, MultiprocessorMixin

class DistanceTransformInputType(enum.Enum):
    '''The input is a binary mask
    
    The distance is computed from the nearest background (false) pixel
    '''
    BinaryMask=1
    
    '''The input is a probability map
    
    The probability map is thresholded and the distance is computed from
    the nearest voxel above threshold.
    '''
    ProbabilityMap=2
    
    '''The input is a segmentation
    
    The segmentation is eroded by one and the distance is computed from the
    nearest unlabeled voxel.
    '''
    Segmentation=3
    
class DistanceTransformTaskMixin:
    
    volume = VolumeParameter(
        description="The volume to be processed")
    input_location = DatasetLocationParameter(
        description="The location of the input dataset")
    input_type = luigi.EnumParameter(
        enum=DistanceTransformInputType,
        description="The type of data being processed")
    output_location = DatasetLocationParameter(
        description="The location for the output dataset")
    #
    # Optional parameters
    #
    invert = luigi.BoolParameter(
        description="Invert the polarity of the input")
    xy_nm = luigi.FloatParameter(
        default=4,
        description="The size of a voxel in the x and y directions")
    z_nm = luigi.FloatParameter(
        default=30,
        description="The size of a voxel in the z direction")
    xy_block_size = luigi.IntParameter(
        default=512,
        description="The size of a sub-block in the x and y directions for the "
        "parallel distance transform")
    z_block_size = luigi.IntParameter(
        default=50,
        description="The size of a sub-block in the z direction for the "
        "parallel distance transform")
    xy_overlap = luigi.IntParameter(
        default=40,
        description="Overlap between blocks in the x and y directions")
    z_overlap = luigi.IntParameter(
        default=5,
        description="Overlap between blocks in the z direction")
    threshold = luigi.IntParameter(
        default=128,
        description="Threshold for binarizing a probability map")
    invert_output = luigi.BoolParameter(
        description="Invert output so that most distant point has a value of 0."
        " The data is clipped at 255 with this option and stored as uint8.")
    scaling_factor = luigi.FloatParameter(
        default=1.0,
        description="Scale the distance by this factor.")
    data_type = luigi.Parameter(
        default="uint8",
        description="The Numpy data type of the output: uint8, uint16 or "
        "uint32")
    
    def input(self):
        yield TargetFactory().get_volume_target(self.input_location,
                                                self.volume)
    
    def output(self):
        return TargetFactory().get_volume_target(self.output_location,
                                                 self.volume)

class DistanceTransformRunMixin:
    
    def ariadne_run(self):
        volume = self.input().next().imread()
        if self.input_type == DistanceTransformInputType.ProbabilityMap:
            volume = volume < self.threshold
        elif self.input_type == DistanceTransformInputType.Segmentation:
            strel = np.array([[[False, False, False],
                               [False, True, False], 
                               [False, False, False]],
                              [[False, True , False],
                               [True,  True,  True],
                               [False, True, False]],
                              [[False, False, False],
                               [False, True, False], 
                               [False, False, False]]])
            erode_segmentation(volume, strel, in_place=True)
            volume = volume == 0
        if self.invert:
            volume = ~volume
        result = parallel_distance_transform(
            volume, self.xy_nm, self.z_nm, self.xy_overlap, self.z_overlap,
            self.xy_block_size, self.z_block_size, self.cpu_count)
        if self.scaling_factor != 1.0:
            result *= self.scaling_factor
        if self.invert_output:
            result = np.max(result) - result
        data_type = getattr(np, self.data_type)
        limit = np.iinfo(data_type).max()
        result[result > limit] = limit
        self.output().imwrite(result.astype(data_type))

class DistanceTransformTask(DistanceTransformTaskMixin,
                            DistanceTransformRunMixin,
                            RequiresMixin,
                            RunMixin,
                            MultiprocessorMixin,
                            luigi.Task):
    '''Compute the distance transform on a volume'''
    
    task_namespace="ariadne_microns_pipeline"