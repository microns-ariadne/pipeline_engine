import enum
import h5py
import luigi
import multiprocessing
import numpy as np
import os
import rh_config
import rh_logger
import shutil
import subprocess
import tempfile

from .neuroproof_common import NeuroproofVersion, NeuroproofDilateMixin
from .neuroproof_common import write_seg_volume, write_prob_volume
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
                         self.gt_loading_plan_path] + \
            list(self.additional_locations)
        for loading_plan in loading_plans:
            for tgt in DestVolumeReader(loading_plan).get_source_targets():
                yield tgt
    
    def output(self):
        return luigi.LocalTarget(self.output_location)


class NeuroproofLearnRunMixin(NeuroproofDilateMixin):
    
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
    neuroproof_version = luigi.EnumParameter(
        enum=NeuroproofVersion,
        default=NeuroproofVersion.MIT,
        description="The command-line convention to be used to run the "
        "Neuroproof binary")
    dilation_xy=luigi.IntParameter(
        default=1,
        description="Amount to dilate the membrane prediction in X and Y")
    dilation_z=luigi.IntParameter(
        default=1,
        description="Amount to dilate the membrane prediction in Z")
    
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
        loading_plan = self.dilate_membrane_prediction()
        prob_target = DestVolumeReader(loading_plan)
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
            #
            # Here, we use multiprocessing to launch the writes. On the surface
            # of it, this looks like we're using multiple processes to do
            # the writes in parallel, but more importantly
            #
            #            NeuroproofLearnTask
            #                /           \
            #       Write processes   Neuroproof
            #
            # the handles opened by the write processes take place in a child
            # process which ends before Neuroproof starts and the HDF5 library
            # should close the handles before Neuroproof starts.
            #
            pool = multiprocessing.Pool(3)
            if self.neuroproof_version != NeuroproofVersion.MINIMAL:
                #
                # neuroproof_graph_predict has a hardcoded dataset name
                #
                dataset_name = "volume/predictions";
                #
                # neuroproof_graph_predict has the predictions in the
                # Ilastik convention: x, y, z, c
                #
                transpose = True
            else:
                #
                # Neuroproof_stack_learn can take any dataset name
                #
                dataset_name = "stack"
                #
                # Neuroproof_stack_learn has the predictions in the standard
                # format: z, y, x, c
                #
                transpose = False
            duplicate = self.neuroproof_version == NeuroproofVersion.MIT
            pred_process = pool.apply_async(
                write_prob_volume, 
                args=[prob_target, additional_map_targets, pred_path, 
                      dataset_name, transpose, duplicate])
            dataset_name = "stack"
            seg_process = pool.apply_async(
                write_seg_volume, 
                args=(watershed_path, seg_target, dataset_name))
            gt_process = pool.apply_async(
                write_seg_volume,
                args=(gt_path, gt_target, dataset_name))
            pool.close()
            pool.join()
            pred_process.get()
            seg_process.get()
            gt_process.get()
            
            if self.neuroproof_version == NeuroproofVersion.MINIMAL:
                #
                # Run the Neuroproof_stack_learn application
                #
                args = [
                    self.neuroproof,
                    "-watershed", watershed_path, dataset_name,
                    "-prediction", pred_path, dataset_name,
                    "-groundtruth", gt_path, dataset_name,
                    "-iteration", str(self.num_iterations),
                    "-strategy", str(self.strategy.value),
                    "-classifier", self.output_location]
                if not self.use_mito:
                    args.append("-nomito")
                if self.prune_feature:
                    args.append("-prune_feature")
            else:
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

            rh_logger.logger.report_event(" ".join(args))
            subprocess.check_call(args, env=env)

        finally:
            try:
                self.delete_dilated_loading_plan(loading_plan)
            except:
                rh_logger.logger.report_exception()
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
