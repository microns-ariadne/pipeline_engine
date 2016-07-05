import luigi
import tempfile

from ..parameters import VolumeParameter, DatasetLocationParameter, \
     DatasetLocation
from ..tasks.mask import MaskBorderTask
from ..tasks.segment import SegmentTask


class ExamplePipeline(luigi.WrapperTask):
    
    volume = VolumeParameter()
    prob_location = DatasetLocationParameter()
    seg_location = DatasetLocationParameter()

    def requires(self):
        temp_location = DatasetLocation(
            [tempfile.gettempdir()], "mask", "{x:04d}_{y:04d}_{z:04d}")
        mask_border_task = MaskBorderTask(
            volume=self.volume,
            prob_location=self.prob_location,
            mask_location=temp_location)
        
        segment_task = SegmentTask(
            volume=self.volume,
            prob_location=self.prob_location,
            mask_location=temp_location,
            seg_location = self.seg_location)
        segment_task.set_requirement(mask_border_task)
        yield segment_task

    task_namespace = "ariadne_microns_pipeline"