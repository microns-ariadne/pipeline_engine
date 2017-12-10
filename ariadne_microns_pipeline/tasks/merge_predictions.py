# A task to merge several predictions into one.
#
import enum
import luigi
import numpy as np

from ariadne_microns_pipeline.targets.volume_target \
     import DestVolumeReader, SrcVolumeTarget
from .utilities import DatasetMixin, RunMixin

class MergeOperation(enum.Enum):
    Average=1
    Minimum=2
    Maximum=3
    
class MergePredictionsTask(DatasetMixin, RunMixin, luigi.Task):
    '''This task merges several predictions into one
    
    '''
    task_namespace="ariadne_microns_pipeline"
    
    loading_plans = luigi.ListParameter(
        description="The loading plans that will be merged")
    operation = luigi.EnumParameter(
        enum=MergeOperation,
        default=MergeOperation.Average,
        description="The operation to perform to merge the volumes")
    invert = luigi.BoolParameter(
        description="Invert the image intensity")
    
    def input(self):
        for loading_plan in self.loading_plans:
            for tgt in DestVolumeReader(loading_plan).get_source_targets():
                yield(tgt)
    
    def ariadne_run(self):
        lp0 = DestVolumeReader(self.loading_plans[0])
        result = lp0.imread()
        if result.dtype == np.uint8 and \
           self.operation == MergeOperation.Average:
            dtype = np.uint8
            result = result.astype(np.uint16)
        else:
            dtype = result.dtype
        for loading_plan in self.loading_plans[1:]:
            volume = DestVolumeReader(loading_plan).imread()\
                .astype(result.dtype)
            if self.operation == MergeOperation.Average:
                result += volume
            elif self.operation == MergeOperation.Maximum:
                mask = volume > result
                result[mask] = volume[mask]
            elif self.operation == MergeOperation.Minimum:
                mask = volume < result
                result[mask] = volume[mask]
        if self.operation == MergeOperation.Average:
            result = (result / len(self.loading_plans)).astype(dtype)
        if self.invert:
            result = np.iinfo(dtype).max - result
        SrcVolumeTarget(self.storage_plan).imwrite(result)