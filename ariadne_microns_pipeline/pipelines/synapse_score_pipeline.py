'''synapse_score_pipeline - a pipeline to score synapse ground-truth

This pipeline scores a synapse classification against a synapse ground-truth
without regard to neurons. A ground-truth synapse can either have no matching
synapse in the detected, a single match. Thus, there are two types of
false positive - a synapse that does not match the ground truth or that
matches doubly.
'''

import json
import luigi
import numpy as np
import os
import rh_logger

from ..parameters import VolumeParameter, Volume, DEFAULT_LOCATION
from ..targets.classifier_target import PixelClassifierTarget
from ..targets.volume_target import write_loading_plan, write_storage_plan
from ..tasks import AMTaskFactory
from ..tasks.classify import ClassifyShimTask
from ..tasks.find_seeds import Dimensionality
from ..tasks.match_synapses import MatchMethod
from ..volumedb import VolumeDB, Persistence, UINT8, UINT16, UINT32
from .pipeline import SYNAPSE_DATASET, SYNAPSE_RECEPTOR_DATASET, \
     SYNAPSE_TRANSMITTER_DATASET, SYN_GT_DATASET, IMG_DATASET, \
     SYN_SEG_DATASET, SYN_SEG_GT_DATASET


class SynapseScorePipelineTask(luigi.Task):
    '''The SynapseScorePipelineTask scores a synapse segmentation against gt
    
    This pipeline runs a synapse classifier and produces a segmentation from
    the classification. It then matches this against the ground-truth
    synapse segmentation, producing a JSON dictionary with the following
    keys:
    
    * true_positives - the # of cases where one detected synapse matches one
                       ground-truth synapse
    
    * false_positives - the # of cases where a detected synapse does not match
                        any ground-truth synapse
    
    * false_negatives - the # of cases where a ground-truth synapse does not
                        match any detected synapse
                        
    * double_synapses - the # of cases where a ground-truth synapse matches more
                        than one detected synapse
    
    * merged_synapses - the # of cases where a detected synapse matches more
                        than one ground-truth synapse (it merges two synapses)
    
    * precision - true positives / total positives
    
    * recall - true positives / total gt
    
    * synapse_match_path - the path to the low-level synapse match table.
    '''
    task_namespace = "ariadne_microns_pipeline"
    #
    # Butterfly parameters
    #
    experiment = luigi.Parameter(
        description="The Butterfly experiment that produced the dataset")
    sample = luigi.Parameter(
        description="The ID of the biological sample that was imaged")
    dataset = luigi.Parameter(
        description="The name of the volume that was imaged")
    channel = luigi.Parameter(
        default="raw",
        description="The name of the channel from which we take data")
    synapse_channel = luigi.Parameter(
        default="synapse",
        description="The name of the channel containing ground truth "
        "synapse data")
    url = luigi.Parameter(
        description="The URL of the Butterfly REST endpoint")
    #
    # Volume parameters
    #
    volume = VolumeParameter(
        description="The volume to segment")
    xy_nm = luigi.FloatParameter(
        default=4.0,
        description="The size of a voxel in the x and y directions")
    z_nm = luigi.FloatParameter(
        default=30.0,
        description="The size of a voxel in the z direction")
    temp_dir = luigi.Parameter(
        description="The root directory for ephemeral and intermediate data")
    root_dir = luigi.Parameter(
        description="The root directory for data to be saved")
    volume_db_url = luigi.Parameter(
        default=DEFAULT_LOCATION,
        description="The sqlalchemy URL to use to connect to the volume "
        "database")
    datatypes_to_keep = luigi.ListParameter(
        default=[],
        description="Names of the datasets (e.g. \"neuroproof\") to store "
        "under the root directory.")
    #
    # The classifier to use. It should either produce a synapse channel
    # or a transmitter and receiver channel
    #
    pixel_classifier_path = luigi.Parameter(
        description="Path to pickled pixel classifier")
    synapse_class_name = luigi.Parameter(
        description="The name of the pixel classifier's synapse class",
        default="synapse")
    transmitter_class_name = luigi.Parameter(
        description="The name of the voxel classifier class that gives "
                    "the probability that a voxel is on the transmitter side "
                    "of a synapse.",
        default="transmitter")
    receptor_class_name = luigi.Parameter(
        description="The name of the voxel classifier class that gives "
                    "the probability that a voxel is on the receptor side "
                    "of a synapse.",
        default="receptor")
    wants_transmitter_receptor_synapse_maps = luigi.BoolParameter(
        description="Use a synapse transmitter and receptor probability map "
                    "instead of a map of synapse voxel probabilities.")
    #
    # Parameters for synapse segmentation
    #
    min_synapse_area = luigi.IntParameter(
        description="Minimum area for a synapse",
        default=1000)
    synapse_xy_sigma = luigi.FloatParameter(
        description="Sigma for smoothing Gaussian for synapse segmentation "
                     "in the x and y directions.",
        default=.5)
    synapse_z_sigma = luigi.FloatParameter(
        description="Sigma for smoothing Gaussian for symapse segmentation "
                     "in the z direction.",
        default=.5)
    synapse_min_size_2d = luigi.IntParameter(
        default=250,
        description="Remove isolated synapse foreground in a plane if "
        "less than this # of pixels")
    synapse_max_size_2d = luigi.IntParameter(
        default=15000,
        description = "Remove large patches of mislabeled synapse in a plane "
        "that have an area greater than this")
    min_synapse_depth = luigi.IntParameter(
        default=4,
        description="Minimum acceptable size of a synapse in the Z direction")
    synapse_threshold = luigi.FloatParameter(
        description="Threshold for synapse voxels vs background voxels",
        default=128.)
    #
    # parameters for synapse statistics
    #
    synapse_match_method = luigi.EnumParameter(
        enum=MatchMethod,
        default=MatchMethod.overlap,
        description="Method for matching detected synapses against "
        "ground-truth synapses")
    synapse_min_overlap_pct = luigi.FloatParameter(
        default=25.0,
        description="The minimum acceptable overlap between "
            "ground-truth and detected synapses")
    synapse_min_gt_overlap_pct = luigi.FloatParameter(
        default=50.0,
        description="The minimum acceptable overlap of the ground truth")
    synapse_max_distance = luigi.FloatParameter(
        default=100.,
        description="The maximum allowed distance between centroids of "
             "ground-truth and detected synapses")
    synapse_gt_classes = luigi.ListParameter(
        default=[],
        description="A list of the values of synapse voxels in the "
        "ground-truth. For instance if 1=pre-synaptic and 2=post-synaptic and "
        "3=gap junctions, use [1, 2] to exclude gap junctions. The default is "
        "to include any value other than zero.")
    synapse_report_path = luigi.Parameter(
        description="Location for the synapse report .json file")
    
    def output(self):
        return luigi.LocalTarget(self.synapse_report_path)
    
    def requires(self):
        if not hasattr(self, "requirements"):
            self.requirements = self.compute_requirements()
        return self.requirements
    
    def compute_requirements(self):
        '''Compute the requirements for the pipeline'''
        try:
            rh_logger.logger.report_event("Assembling pipeline")
        except:
            rh_logger.logger.start_process(
                "Synapse scoring pipeline",
                "Assembling pipeline")
            #
            # Configuration turns off the luigi-interface logger
            #
        import logging
        logging.getLogger("luigi-interface").disabled = False
        
        self.datasets = {}
        self.tasks = []
        #
        # Prep work
        #
        self.init_db()
        self.factory = AMTaskFactory(self.volume_db_url, 
                                             self.volume_db)
        rh_logger.logger.report_event(
                    "Loading pixel classifier")
        self.pixel_classifier = PixelClassifierTarget(
                    self.pixel_classifier_path)
        
        #
        # Step 1: download image volume
        #
        self.get_image_volume_task()
        #
        # Step 2: download the ground truth
        #
        self.get_gt_volume_task()
        #
        # Step 3: classify the image volume
        #
        self.get_classify_task()
        #
        # Step 4: segment the synapse classification
        #
        self.get_synapse_segmentation_task()
        #
        # Step 5: segment the ground-truth
        # 
        self.get_synapse_gt_segmentation_task()
        #
        # Step 6: match the synapses
        #
        self.get_synapse_match_task()
        #
        # Do the VolumeDB computation
        #
        rh_logger.logger.report_event("Computing load/store plans")
        self.volume_db.compute_subvolumes()
        #
        # Write the loading plans
        #
        rh_logger.logger.report_event("Writing loading plans")
        for loading_plan_id in self.volume_db.get_loading_plan_ids():
            loading_plan_path = self.volume_db.get_loading_plan_path(
                loading_plan_id)
            directory = os.path.dirname(loading_plan_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            write_loading_plan(loading_plan_path, self.volume_db, 
                              loading_plan_id)
        #
        # Write the storage plans
        #
        rh_logger.logger.report_event("Writing storage plans")
        for dataset_id in self.volume_db.get_dataset_ids():
            write_storage_plan(self.volume_db, dataset_id)
        #
        # Hook up dependencies.
        #
        dependentd = dict([(_, [] ) for _ in self.datasets])
        for task in self.tasks:
            for tgt in task.input():
                path = tgt.path
                if path in self.datasets:
                    task.set_requirement(self.datasets[path])
                    dependentd[path].append(task)
        return [self.synapse_match_task]

    def register_datatype(self, name, datatype, doc):
        '''Register a datatype with the VolumeDB
        
        :param name: the name of the datatype, e.g. "neuroproof"
        :param datatype: the Numpy datatype of the dataset, e.g. "uint8"
        :param doc: a short description of what the datatype is
        '''
        if name in self.datatypes_to_keep:
            persistence = Persistence.Permanent
        else:
            persistence = Persistence.Temporary
        self.volume_db.register_dataset_type(name, persistence, datatype, doc)
        
    def init_db(self):
        '''Initialize the volume DB'''
        if self.volume_db_url == DEFAULT_LOCATION:
            self.volume_db_url = "sqlite:///%s/volume.db" % self.root_dir
            if not os.path.isdir(self.root_dir):
                os.makedirs(self.root_dir)
        rh_logger.logger.report_event("Creating volume DB at URL %s" %
                                      self.volume_db_url)
        self.volume_db = VolumeDB(self.volume_db_url, "w")
        #
        # Register the temp and root directories
        #
        self.volume_db.set_target_dir(self.root_dir)
        self.volume_db.set_temp_dir(self.temp_dir)
        #
        # Define the datatypes
        #
        rh_logger.logger.report_event("Creating data types in the volume DB")
        self.register_datatype(
            SYN_SEG_DATASET, UINT16,
            "A segmentation of a synapse")
        self.register_datatype(IMG_DATASET, UINT8,
                              "The initial raw image data")
        self.register_datatype(SYNAPSE_DATASET, UINT8,
                               "Probability of a voxel being in a synapse")
        self.register_datatype(SYNAPSE_TRANSMITTER_DATASET, UINT8,
                               "Probability of a voxel being in the "
                               "presynaptic partner of a synapse")
        self.register_datatype(SYNAPSE_RECEPTOR_DATASET, UINT8,
                               "Probability of a voxel being in the "
                               "postsynaptic partner of a synapse")
        self.register_datatype(SYN_GT_DATASET, UINT8, 
                               "The markup for synapse ground-truth")
        self.register_datatype(SYN_SEG_GT_DATASET, UINT32,
                               "The ground-truth synapse segmentation")
        
    def get_image_volume_task(self):
        #
        # Compute the dimensions for the classifier input
        #
        classifier = self.pixel_classifier.classifier
        self.nn_x_pad = classifier.get_x_pad()
        self.nn_y_pad = classifier.get_y_pad()
        self.nn_z_pad = classifier.get_z_pad()
        self.cx0 = self.volume.x - self.nn_x_pad
        self.cx1 = self.volume.x1 + self.nn_x_pad
        self.cy0 = self.volume.y - self.nn_y_pad
        self.cy1 = self.volume.y1 + self.nn_y_pad
        self.cz0 = self.volume.z - self.nn_z_pad
        self.cz1 = self.volume.z1 + self.nn_z_pad
        
        self.img_volume = Volume(
            self.cx0, self.cy0, self.cz0,
            self.cx1-self.cx0, self.cy1 - self.cy0, self.cz1 - self.cz0)
        self.butterfly_task = self.factory.gen_get_volume_task(
            experiment = self.experiment,
            sample = self.sample,
            dataset = self.dataset,
            channel = self.channel,
            url = self.url,
            volume = self.img_volume,
            dataset_name=IMG_DATASET)
        self.tasks.append(self.butterfly_task)
        self.datasets[self.butterfly_task.output().path] = self.butterfly_task
    
    def get_gt_volume_task(self):
        self.gt_butterfly_task = self.factory.gen_get_volume_task(
            experiment = self.experiment,
            sample = self.sample,
            dataset = self.dataset,
            channel = self.synapse_channel,
            url = self.url,
            volume = self.volume,
            dataset_name=SYN_GT_DATASET)
        self.tasks.append(self.gt_butterfly_task)
        self.datasets[self.gt_butterfly_task.output().path] = \
            self.gt_butterfly_task
    
    def get_classify_task(self):
        if self.wants_transmitter_receptor_synapse_maps:
            datasets = {
                self.transmitter_class_name: SYNAPSE_TRANSMITTER_DATASET,
                self.receptor_class_name: SYNAPSE_RECEPTOR_DATASET }
        else:
            datasets = {
                self.synapse_class_name: SYNAPSE_DATASET }
        self.classifier_task = self.factory.gen_classify_task(
            datasets=datasets,
            img_volume=self.img_volume,
            output_volume=self.volume,
            dataset_name=IMG_DATASET,
            classifier_path = self.pixel_classifier_path)
        self.tasks.append(self.classifier_task)
        for channel in datasets.values():
            shim_task = ClassifyShimTask.make_shim(
                classify_task=self.classifier_task,
                dataset_name=channel)
            self.datasets[shim_task.output().path] = shim_task
            self.tasks.append(shim_task)
    
    def get_synapse_segmentation_task(self):
        if self.wants_transmitter_receptor_synapse_maps:
            self.synapse_task = self.factory.gen_find_synapses_tr_task(
                volume=self.volume,
                transmitter_dataset_name=SYNAPSE_TRANSMITTER_DATASET,
                receptor_dataset_name=SYNAPSE_RECEPTOR_DATASET,
                output_dataset_name=SYN_SEG_DATASET,
                threshold=self.synapse_threshold,
                erosion_xy=0, # unused parameter
                erosion_z=0, # unused parameter
                sigma_xy=self.synapse_xy_sigma,
                sigma_z=self.synapse_z_sigma,
                min_size_2d=self.synapse_min_size_2d,
                max_size_2d=self.synapse_max_size_2d,
                min_size_3d=self.min_synapse_area,
                min_slice=self.min_synapse_depth)
        else:
            self.synapse_task = self.factory.gen_find_synapses_task(
                volume=self.volume,
                synapse_prob_dataset_name=SYNAPSE_DATASET,
                output_dataset_name=SYN_SEG_DATASET,
                threshold=self.synapse_threshold,
                erosion_xy=0,
                erosion_z=0,
                sigma_xy=self.synapse_xy_sigma,
                sigma_z=self.synapse_z_sigma,
                min_size_2d=self.synapse_min_size_2d,
                max_size_2d=self.synapse_max_size_2d,
                min_size_3d=self.min_synapse_area,
                min_slice=self.min_synapse_depth)
        self.datasets[self.synapse_task.output().path] = self.synapse_task
        self.tasks.append(self.synapse_task)
    
    def get_synapse_gt_segmentation_task(self):
        self.synapse_gt_seg_task = self.factory.gen_cc_segmentation_task(
            volume=self.volume,
            prob_dataset_name=SYN_GT_DATASET,
            seg_dataset_name = SYN_SEG_GT_DATASET,
            threshold=0,
            dimensionality=Dimensionality.D3,
            fg_is_higher=True)
        self.synapse_gt_seg_task.classes = self.synapse_gt_classes
        self.datasets[self.synapse_gt_seg_task.output().path] =\
            self.synapse_gt_seg_task
        self.tasks.append(self.synapse_gt_seg_task)
    
    def get_synapse_match_task(self):
        synapse_match_location = os.path.join(
            self.temp_dir, "synapse-match.json")
        self.synapse_match_task = self.factory.gen_match_synapses_task(
            volume=self.volume,
            gt_dataset_name=SYN_SEG_GT_DATASET,
            detected_dataset_name=SYN_SEG_DATASET,
            output_location=synapse_match_location,
            method=self.synapse_match_method)
        self.synapse_match_task.min_overlap_pct = \
            self.synapse_min_overlap_pct
        self.synapse_match_task.min_gt_overlap_pct = \
            self.synapse_min_gt_overlap_pct
        self.synapse_match_task.max_distance = self.synapse_max_distance
        self.tasks.append(self.synapse_match_task)
    
    def run(self):
        with self.synapse_match_task.output().open() as fd:
            dataset = json.load(fd)
        
        gt_labels = np.array(dataset["gt_labels"])
        detected_labels = np.array(dataset["detected_labels"])
        detected_per_gt = np.array(dataset["detected_per_gt"])
        gt_per_detected = np.array(dataset["gt_per_detected"])
        
        detected_counts = np.bincount(
            gt_per_detected, minlength = np.max(detected_labels) + 1)[
                detected_labels]
        gt_counts = np.bincount(
            detected_per_gt, minlength = np.max(gt_labels) + 1) [
                gt_labels]
        
        double_synapses = np.sum(gt_counts[gt_counts >= 2] - 1)
        merged_synapses = np.sum(detected_counts[detected_counts >= 2] - 1)
        true_positives = np.sum(gt_counts > 0) - double_synapses
        total_positives = len(detected_counts)
        total_gt = len(gt_counts)
        precision = float(true_positives) / total_positives
        recall = float(true_positives) / total_gt
        d = dict(synapse_match_path=self.synapse_match_task.output().path,
                 true_positives=true_positives,
                 false_positives = total_positives - true_positives,
                 false_negatives = total_gt - true_positives,
                 total_gt = total_gt,
                 total_positives=total_positives,
                 double_synapses=double_synapses,
                 merged_synapses=merged_synapses,
                 precision=precision,
                 recall=recall)
        with self.output().open("w") as fd:
            json.dump(d, fd)