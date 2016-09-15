'''The ariadne_microns_pipeline.tasks package holds all of the Ariadne tasks

The organization is the following: there are tasks and task mixins. The tasks
declare the inputs and output and parameterizations and the task mixins have
an `ariadne_run()` method that operates within the context of the tasks.

On top of the whole thing, there is a mixin for local execution and mixins
for cluster execution that connect "run()" to "ariadne_run()" and a
mapper/factory that lets you select the task class for a given task and
execution mechanism.
'''

from .block import BlockTask
from .classify import ClassifyTask
from .connected_components import \
     AllConnectedComponentsTask, ConnectedComponentsTask, VolumeRelabelingTask
from .download_from_butterfly import DownloadFromButterflyTask
from .factory import AMTaskFactory
from .find_seeds import FindSeedsTask, Dimensionality, SeedsMethodEnum
from .mask import MaskBorderTask
from .neuroproof import NeuroproofTask
from .nplearn import NeuroproofLearnTask
from .segment import SegmentTask, SegmentCC2DTask, UnsegmentTask
from .skeletonize import SkeletonizeTask
from .visualize import VisualizeTask, PipelineVisualizeTask

all = [AllConnectedComponentsTask, BlockTask, ClassifyTask, 
       ConnectedComponentsTask,
       DownloadFromButterflyTask, AMTaskFactory,
       FindSeedsTask, Dimensionality, SeedsMethodEnum,
       MaskBorderTask, NeuroproofTask, NeuroproofLearnTask, 
       SegmentTask, SegmentCC2DTask, UnsegmentTask,
       VisualizeTask, VolumeRelabelingTask, PipelineVisualizeTask, 
       SkeletonizeTask]
