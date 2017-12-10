from .boss_pipeline import BossPipelineTask
from .classify_pipeline import ClassifyPipelineTask
from .pipeline import PipelineTask
from .stitch_pipeline import StitchPipelineTask
from merge_predictions_pipeline import MergePredictionsPipeline


all=[BossPipelineTask, ClassifyPipelineTask, PipelineTask, StitchPipelineTask,
     MergePredictionsPipeline]