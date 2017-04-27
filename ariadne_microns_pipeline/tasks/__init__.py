'''The ariadne_microns_pipeline.tasks package holds all of the Ariadne tasks

The organization is the following: there are tasks and task mixins. The tasks
declare the inputs and output and parameterizations and the task mixins have
an `ariadne_run()` method that operates within the context of the tasks.

On top of the whole thing, there is a mixin for local execution and mixins
for cluster execution that connect "run()" to "ariadne_run()" and a
mapper/factory that lets you select the task class for a given task and
execution mechanism.
'''

from .classify import ClassifyTask
from .connected_components import \
     AllConnectedComponentsTask, ConnectedComponentsTask, VolumeRelabelingTask,\
     StoragePlanRelabelingTask
from .copy import CopyFileTask, CopyStoragePlan, DeleteStoragePlan
from .copy import BossShardingTask
from .connect_synapses import ConnectSynapsesTask
from .distance_transform import DistanceTransformTask
from .download_from_butterfly import DownloadFromButterflyTask
from .download_from_butterfly import LocalButterflyTask
from .factory import AMTaskFactory
from .find_seeds import FindSeedsTask, Dimensionality, SeedsMethodEnum
from .mask import MaskBorderTask
from .match_neurons import MatchNeuronsTask
from .match_synapses import MatchSynapsesTask
from .neuroproof import NeuroproofTask
from .nplearn import NeuroproofLearnTask
from .segment import SegmentTask, SegmentCC2DTask, UnsegmentTask
from .skeletonize import SkeletonizeTask
from .stitch_segmentation import StitchSegmentationTask
from .synapse_statistics import SynapseStatisticsTask

all = [AllConnectedComponentsTask, BossShardingTask, ClassifyTask, 
       ConnectedComponentsTask, ConnectSynapsesTask,
       CopyFileTask, CopyStoragePlan, DeleteStoragePlan,
       DistanceTransformTask,
       DownloadFromButterflyTask, AMTaskFactory,
       FindSeedsTask, Dimensionality, SeedsMethodEnum,
       MaskBorderTask, MatchNeuronsTask, MatchSynapsesTask,
       NeuroproofTask, NeuroproofLearnTask, 
       SegmentTask, SegmentCC2DTask, UnsegmentTask,
       VolumeRelabelingTask, StoragePlanRelabelingTask,
       SkeletonizeTask, StitchSegmentationTask, SynapseStatisticsTask]
