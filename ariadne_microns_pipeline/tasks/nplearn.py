import enum
import h5py
import luigi
import numpy as np
import os
import rh_config
import rh_logger
import shutil
import subprocess
import tempfile

from .utilities import RequiresMixin, RunMixin, CILKCPUMixin
from ..targets import DestVolumeReader

class StrategyEnum(enum.Enum):
    misclassified=1
    all=2
    LASH=3
    random_learning=4
    uncertain_learning=5
    semi_supervised_learning=6
    IWAL_learning=7
    co_training=8
    simulating=9

class NeuroproofLearnTaskMixin:
    prob_loading_plan_path = luigi.Parameter(
        description="Location of the probability prediction volume")
    additional_locations = luigi.ListParameter(
        default=[],
        description="Additional probability map locations for Neuroproof")
    seg_loading_plan_path = luigi.Parameter(
        description="Location of the pipeline's watershed segmentation")
    gt_loading_plan_path = luigi.Parameter(
        description="Location of the ground truth segmentation")
    output_location = luigi.Parameter(
        description="Location for the classifier file. Use an .xml extension "
        "to use the OpenCV random forest classifier. Use an .h5 extension "
        "to use the Vigra random forest classifier")
    
    def input(self):
        loading_plans = [self.prob_loading_plan_path,
                         self.seg_loading_plan_path,
                         self.gt_loading_plan_path] + self.additional_locations
        for loading_plan in loading_plans:
            for tgt in DestVolumeReader(loading_plan).get_source_targets():
                yield tgt
    
    def output(self):
        return luigi.LocalTarget(self.output_location)


class NeuroproofLearnRunMixin:
    
    neuroproof = luigi.Parameter(
        description="Location of the neuroproof_graph_learn binary")
    neuroproof_ld_library_path = luigi.Parameter(
        description="Library paths to Neuroproof's shared libraries. "
        "This should include paths to CILK stdc++ libraries, Vigra libraries, "
        "JSONCPP libraries, and OpenCV libraries.")
    strategy = luigi.EnumParameter(
        enum=StrategyEnum,
        default=StrategyEnum.all,
        description="Learning strategy to use")
    num_iterations = luigi.IntParameter(
        description="Number of iterations used for learning")
    prune_feature = luigi.BoolParameter(
        description="Automatically prune useless features")
    use_mito = luigi.BoolParameter(
        description="Set delayed mito agglomeration")
    
    def ariadne_run(self):
        '''Run neuroproof_graph_learn in a subprocess'''
        #
        # Create two .h5 files in a temporary directory:
        #
        # pred.h5 contains the following datasets:
        #      volume/predictions - the membrane predictions as a 4d array
        #                           of channel, z, y, and x
        #      stack - a 4d array of channel, z, y and x
        #
        # gt.h5 contains a stack dataset
        #
        prob_target = DestVolumeReader(self.prob_loading_plan_path)
        seg_target = DestVolumeReader(self.seg_loading_plan_path)
        gt_target = DestVolumeReader(self.gt_loading_plan_path)
        additional_map_targets = [DestVolumeReader[_]
                                  for _ in self.additional_locations]
        task_name = self.task_name()
        tempdir = tempfile.mkdtemp()
        rh_logger.logger.report_event(
            "%s: tempdir = %s" % (task_name, tempdir))
        try:
            pred_path = os.path.join(tempdir, "pred.h5")
            watershed_path = os.path.join(tempdir, "watershed.h5")
            gt_path = os.path.join(tempdir, "gt.h5")
            
            prob_volume = prob_target.imread().astype(np.float32) / 255.
            prob_volume = [prob_volume, prob_volume]
            for tgt in additional_map_targets:
                prob_volume.append(tgt.imread().astype(np.float32) / 255.)
            prob_volume = np.array(prob_volume)
            prob_volume = prob_volume.transpose(3, 2, 1, 0)
            rh_logger.logger.report_event("%s: writing pred.h5" % task_name)
            with h5py.File(pred_path, "w") as fd:
                fd.create_dataset("volume/predictions", data=prob_volume)
                del prob_volume
            with h5py.File(watershed_path, "w") as fd:
                seg_volume = seg_target.imread().astype(np.int32)
                fd.create_dataset("stack", data=seg_volume)
                del seg_volume
            gt_volume = gt_target.imread().astype(np.int32)
            with h5py.File(gt_path, "w") as fd:
                fd.create_dataset("stack", data=gt_volume)
            del gt_volume
            #
            # Run the neuroproof_graph_learn task
            #
            # See neuroproof_graph_learn.cpp for parameters
            #
            args = [
                self.neuroproof,
                '--classifier-name', self.output_location,
                '--strategy-type', str(self.strategy.value),
                '--num-iterations', str(self.num_iterations),
                '--prune_feature', ("1" if self.prune_feature else "0"),
                '--use_mito', ("1" if self.use_mito else "0"),
                '--watershed-file', watershed_path,
                '--prediction-file', pred_path,
                '--groundtruth-file', gt_path
                ]
            #
            # Inject the custom LD_LIBRARY_PATH into the subprocess environment
            #
            env = os.environ.copy()
            if "LD_LIBRARY_PATH" in env:
                ld_library_path = self.neuroproof_ld_library_path + os.pathsep +\
                    env["LD_LIBRARY_PATH"]
            else:
                ld_library_path = self.neuroproof_ld_library_path
            env["LD_LIBRARY_PATH"] = ld_library_path
            #
            # Execute. There's often some wierdness with HDF5 where
            # file handles are inherited by the child process and sad
            # things happen. So I experimented with isolation until I came
            # up with the formula below.
            #
            args = args[0] + ' "' + '" "'.join(args[1:]) + '"'
            subprocess.check_call(args, env=env, shell=True)

        finally:
            try:
                shutil.rmtree(tempdir)
            except:
                #
                # If we can't delete the tempfiles, report it
                # but otherwise eat the exception and leave a mess
                # on disk.
                #
                rh_logger.logger.report_exception()
                


class NeuroproofLearnTask(
    NeuroproofLearnTaskMixin,
    NeuroproofLearnRunMixin,
    RequiresMixin,
    RunMixin,
    CILKCPUMixin,
    luigi.Task):
    '''Train a Neuroproof classifier
    
    This task runs neuroproof_graph_learn to produce a classifier. The inputs
    are the membrane probabilities that you will use for the Neuroproof
    task and the ground truth segmentation for the volume.
    '''
    
    task_namespace = "ariadne_microns_pipeline"