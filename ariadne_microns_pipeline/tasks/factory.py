'''The task factory creates tasks for particular pipeline steps.'''

import contextlib
import json
import luigi
import os
import rh_config

from .download_from_butterfly import DownloadFromButterflyTask
from .download_from_butterfly import LocalButterflyTask
from .classify import ClassifyTask
from .connected_components import AllConnectedComponentsTask
from .connected_components import ConnectedComponentsTask
from .connected_components import FakeAllConnectedComponentsTask
from .connected_components import VolumeRelabelingTask
from .connected_components import StoragePlanRelabelingTask
from .connected_components import JoinConnectedComponentsTask
from .connect_synapses import ConnectSynapsesTask, SynapseRelabelingTask
from .connect_synapses import AggregateSynapseConnectionsTask
from .copytasks import BossShardingTask, CopyStoragePlanTask,\
     CopyLoadingPlanTask, ChimericSegmentationTask, AggregateLoadingPlansTask
from .distance_transform import DistanceTransformInputType
from .distance_transform import DistanceTransformTask
from .filter import FilterSegmentationTask
from .find_seeds import FindSeedsTask, Dimensionality, SeedsMethodEnum
from .find_synapses import FindSynapsesTask
from .json_to_csv_task import JSONToCSVTask
from .mask import MaskBorderTask
from .match_neurons import MatchNeuronsTask
from .match_synapses import MatchSynapsesTask
from .neuroproof import NeuroproofTask
from .neuroproof_common import NeuroproofVersion
from .neuroproof_stitch import NeuroproofStitchTask
from .nplearn import NeuroproofLearnTask, StrategyEnum
from .segment import \
     SegmentTask, SegmentCC2DTask, SegmentCC3DTask, UnsegmentTask, \
     ZWatershedTask, WaterZTask
from .segmentation_statistics import \
     SegmentationStatisticsTask, SegmentationReportTask
from .skeletonize import SkeletonizeTask
from .stitch_segmentation import StitchSegmentationTask, Compression
from .synapse_statistics import SynapseStatisticsTask
from .utilities import to_hashable
from ..parameters import EMPTY_LOCATION
from ..volumedb import VolumeDB, get_storage_plan_path, get_loading_plan_path
from ..targets import SrcVolumeTarget, DestVolumeReader

class AMTaskFactory(object):
    '''Factory for creating Ariadne/Microns tasks
    
    Each method has its output target first and input targets following,
    e.g.
    
    def gen_foo_from_bar_and_baz(foo, bar, baz)
    
    The return value is the task that performs the action
    '''
    
    def __init__(self, volume_db_url, volume_db):
        '''Initialize the task factory
        
        :param volume_db_url: location of the volume DB for this pipeline. A
        URL suitable for a sqlalchemy engine.
        :param volume_db: the actual volume database, open for writing
        '''
        self.volume_db_url = volume_db_url
        assert isinstance(volume_db, VolumeDB)
        self.volume_db = volume_db
    
    class __LP(object):
        '''Class to handle context for creating a loading plan.
        
        See loading_plan() for documentation.
        '''
        def __init__(self, volume_db, volume, loading_plan_id, dataset_name,
                     src_task):
            assert isinstance(volume_db, VolumeDB)
            self.volume_db = volume_db
            self.volume = volume
            self.loading_plan_id = loading_plan_id
            self.dataset_name = dataset_name
            self.src_task = src_task
        def __call__(self, task):
            self.volume_db.register_dataset_dependent(
                self.loading_plan_id, task, self.dataset_name, self.volume,
                src_task=self.src_task)
            return task
    
    def loading_plan(self, volume, dataset_name, src_task=None):
        '''Register a loading plan for a given volume
        
        There are several steps to regisetering a loading plan - getting
        the loading_plan_id, figuring out where to store the loading plan file
        and then actually registering the loading plan. This method is
        designed to be used in the following way:
        
        loading_plan_path, lp = self.loading_plan(volume, dataset_name)
        
        task = lp | MyTask(loading_plan_path=loading_plan_path)
        
        :param volume: the volume to be loaded
        :param dataset_name: the name of the dataset being requested
        :param src_task: the task that produced the dataset. Default is accept
        data from any task
        :returns: a two-tuple of the path to the loading plan and a class
        instance which will register the loading plan when the task is
        piped into it.
        '''
        loading_plan_id = \
            self.volume_db.find_loading_plan_id_by_type_and_volume(
                dataset_name, volume, src_task)
        if loading_plan_id is not None:
            magic_obj = self.__Null()
        else:
            loading_plan_id = self.volume_db.get_loading_plan_id()
            magic_obj = self.__LP(
                self.volume_db, volume, loading_plan_id,  dataset_name, 
                src_task)
        root = self.volume_db.get_datatype_root(dataset_name)
        loading_plan_path = get_loading_plan_path(
            root, loading_plan_id, volume, 
            dataset_name)
        return loading_plan_path, magic_obj
        
    class __SP(object):
        '''Class to handle context for creating a storage plan.
        
        See storage_plan() for documentation.
        '''
        def __init__(self, volume_db, volume, dataset_id, dataset_name):
            assert isinstance(volume_db, VolumeDB)
            self.volume_db = volume_db
            self.volume = volume
            self.dataset_id = dataset_id
            self.dataset_name = dataset_name
        def __call__(self, task):
            self.volume_db.register_dataset(
                self.dataset_id, task, self.dataset_name, self.volume)
            return task
    
    def storage_plan(self, volume, dataset_name):
        '''Register a storage plan for a given volume
        
        There are several steps to regisetering a storage plan - getting
        the dataset_id, figuring out where to store the storage plan file
        and then actually registering the storage plan. This method is
        designed to be used in the following way:
        
        storage_plan_path, sp = self.storage_plan(volume, dataset_name)
        
        task = sp | MyTask(storage_plan=storage_plan_path)
        
        :param volume: the volume to be written
        :param dataset_name: the name of the dataset being written
        :returns: a two-tuple of the path to the storage plan and a class
        instance which will register the storage plan when the task is
        piped into it.
        '''
        dataset_id = self.volume_db.get_dataset_id()
        root = self.volume_db.get_datatype_root(dataset_name)
        storage_plan_path = get_storage_plan_path(
            root, dataset_id, volume, 
            dataset_name)
        return storage_plan_path, self.__SP(
            self.volume_db, volume, dataset_id,  dataset_name)
    
    class __Null(object):
        '''This class can be used in place of __LP or __SP to do nothing
        
        Usage:
        
        null = self.__Null()
        
        foo = null | bar
        
        assert foo == bar
        '''
        def __call__(self, other):
            return other

        
    def gen_get_volume_task(self,
                            experiment,
                            sample,
                            dataset,
                            channel,
                            url,
                            volume,
                            dataset_name,
                            resolution=0):
        '''Get a 3d volume
        
        :param experiment: the experiment done to produce the sample
        :param sample: the sample ID of the tissue that was imaged
        :param dataset: the volume that was imaged
        :param channel: the channel supplying the pixel values
        :param url: the URL of the butterfly server
        :param volume: the volume to fetch
        :type volume: :py:class:`ariadne_microns_pipeline.parameters.Volume`
        :param dataset_name: the name of the dataset's type, e.g. "image"
        :param resolution: the MIPMAP resolution of the volume
        :returns: A task that outputs a volume target.
        '''
        storage_plan, sp = self.storage_plan(volume, dataset_name)
        task = sp(DownloadFromButterflyTask(
            experiment=experiment,
            sample=sample,
            dataset=dataset,
            channel=channel,
            url=url,
            volume=volume,
            resolution=resolution,
            storage_plan=storage_plan))
        return task
    
    def gen_local_butterfly_task(self, index_file, volume, dataset_name):
        '''Get a 3d volume using the LocalButterflyTask
        
        :param index_file: the index file that tells us how to get
                           the tiles
        :param dataset_name: the name of the target dataset
        :param volume: the volume of the target dataset
        '''
        storage_plan, sp = self.storage_plan(volume, dataset_name)
        task = sp(LocalButterflyTask(index_file=index_file,
                                     storage_plan=storage_plan))
        return task

    def gen_classify_task(
        self, datasets, img_volume, output_volume, dataset_name, 
        classifier_path, environment_id, src_task=None):
        '''Classify a volume

        :param datasets: a dictionary with keys of the class indexes or names
             produced by the classifier and values of the names of the
             datasets to be stored (not all datasets from the classifier need
             be stored)
        :param img_volume: the voxel volume to be classified - the volume
        for the *input* of the classifier.
        :param output_volume: the voxel volume after classification, likely
        cropped at the borders.
        :param dataset_name: the name of the dataset type, e.g. "image"
        :param classifier_path: path to a pickled classifer
        :param environment_id: the ID of the worker environment to use
        for this classifier.
        :param src_task: preferred source for image data
        '''
        datasets = to_hashable(datasets)
        #
        # Make the loading plan
        #
        loading_plan, lp = self.loading_plan(img_volume, dataset_name, src_task)
        #
        # Make a storage plan for each output channel
        #
        prob_plans = { }
        prob_dataset_ids = {}
        for prob_dataset_name in datasets.values():
            root = self.volume_db.get_datatype_root(prob_dataset_name)
            dataset_id = self.volume_db.get_dataset_id()
            prob_dataset_ids[prob_dataset_name] = dataset_id
            storage_plan = get_storage_plan_path(
                root, dataset_id, output_volume, prob_dataset_name)
            prob_plans[prob_dataset_name] = storage_plan
        #
        # Cons up a done-file name
        #
        done_file_location = os.path.join(
            self.volume_db.target_dir, str(output_volume.x), 
            str(output_volume.y), str(output_volume.z),
            "%09d-%09d_%09d-%09d_%09d-%09d_%s.done" %
            (output_volume.x, output_volume.x1,
             output_volume.y, output_volume.y1,
             output_volume.z, output_volume.z1,
             "-".join(datasets.values())))
        #
        # Create the task
        #
        task = lp(ClassifyTask(classifier_path=classifier_path,
                               image_loading_plan=loading_plan,
                               prob_plans=to_hashable(prob_plans),
                               class_names=datasets,
                               done_file=done_file_location,
                               environment_id=environment_id))
        #
        # Register the datasets
        #
        for prob_dataset_name in datasets.values():
            self.volume_db.register_dataset(
                prob_dataset_ids[prob_dataset_name], task, prob_dataset_name,
                output_volume)
        return task
    
    def gen_find_seeds_task(
        self, volume, prob_dataset_name, seeds_dataset_name,
        dimensionality=Dimensionality.D3, method=SeedsMethodEnum.Smoothing,
        sigma_xy=3, sigma_z=.4, threshold=1, minimum_distance_xy=5,
        minimum_distance_z=1.5, distance_threshold=5):
        '''Generate a seed finding task
        
        This task produces seeds for watershedding.
        
        :param volume: the volume in which to find the seeds
        :param prob_dataset_name: the name of the probability map, 
                                  e.g. "membrane"
        :param seeds_dataset_name: the name of the output dataset
        :param dimensionality: whether to find seeds per-plane or globally
             within the 3d space
        :param method: the algorithm to use to find seeds
        :param sigma_xy: the smoothing standard deviation in the X and Y
            direction
        :param sigma_z: the smoothing standard deviation in the Z direction
        :param threshold: the threshold cutoff for probabilities
        :param minimum_distance: the minimum distance between seeds in the
            X and Y directions
        :param distance_threshold: the minimum distance from the membrane
            of a seed, for the distance method.
        '''
        #
        # Get the loading plan for the probability map
        #
        loading_plan_path, lp = self.loading_plan(volume, prob_dataset_name)
        #
        # Get the storage plan for the seeds
        #
        storage_plan, sp = self.storage_plan(volume, seeds_dataset_name)
        #
        # Create the task
        #
        task = lp(sp(FindSeedsTask(
            prob_loading_plan_path=loading_plan_path,
            storage_plan=storage_plan,
            dimensionality=dimensionality,
            method=method,
            sigma_xy=sigma_xy,
            sigma_z=sigma_z,
            threshold=threshold,
            minimum_distance_xy=minimum_distance_xy,
            minimum_distance_z=minimum_distance_z,
            distance_threshold=distance_threshold)))
        return task
    
    def gen_segmentation_task(
        self, volume, prob_dataset_name, seeds_dataset_name, 
        mask_dataset_name, seg_dataset_name,
        sigma_xy, sigma_z, dimensionality,
        seeds_src_task=None, mask_src_task=None):
        '''Generate a segmentation task

        Generate a segmentation task.  The task takes a probability map of
        the membranes and a mask of areas to exclude from segmentation. It
        smooths the membrane probabilities with an anisotropic Gaussian
        (different sigmas in XY and Z), thresholds to get
        the seeds for the watershed, then performs a 3d watershed.

        :param volume: the volume to be segmented in global coordinates
        :param prob_dataset_name: The name of the probability dataset,
                                  e.g. "membrane"
        :param mask_dataset_name: The name of the mask dataset, e.g. "mask"
        :param seeds_dataset_name: The name of the seeds dataset
        :param seg_dataset_name: the name of the output segmentation
        :param sigma_xy: the sigma of the smoothing gaussian in the X and Y
        directions
        :param sigma_z: the sigma of the smoothing gaussian in the Z direction
        :param dimensionality: Whether to do 2D or 3D segmentation
        :param seeds_src_task: the task that produced the seeds. Default is
        get seeds from whatever volume overlapped this one.
        :param mask_src_task: the task that produced the mask. Default is get
        mask from any mask that overlaps.
        '''
        #
        # Get the load plans for the inputs
        #
        prob_load_plan_path, plp = self.loading_plan(volume, prob_dataset_name)
        mask_load_plan_path, mlp = self.loading_plan(
            volume, mask_dataset_name, mask_src_task)
        seeds_load_plan_path, slp = self.loading_plan(
            volume, seeds_dataset_name, seeds_src_task)
        #
        # Get the storage plan for the segmentation
        #
        storage_plan, sp = self.storage_plan(volume, seg_dataset_name)
        #
        # Create the task
        #
        task = sp ( plp ( mlp ( slp ( SegmentTask(
            prob_loading_plan_path=prob_load_plan_path,
            mask_loading_plan_path=mask_load_plan_path,
            seed_loading_plan_path=seeds_load_plan_path,
            storage_plan=storage_plan,
            sigma_xy=sigma_xy,
            sigma_z=sigma_z,
            dimensionality=dimensionality)))))
        return task
    
    def gen_cc_segmentation_task(
        self, volume, prob_dataset_name, seg_dataset_name,
        threshold,
        mask_dataset_name = None,
        dimensionality=Dimensionality.D2,
        fg_is_higher=False):
        '''Generate a 2d segmentation task
        
        Generate a 2D segmentation task that performs connected components
        on the individual planes.
        
        :param volume: the volume to be segmented, in global coordinates
        :param prob_dataset_name: Name of the membrane probability dataset,
        e.g. "membrane"
        :param mask_dataset_name: Name of the mask dataset, e.g. "mask"
        :param seg_dataset_name: Name for the segmentation dataset
        :param threshold: the cutoff in the membrane probabilities between
        membrane and not-membrane, scaled from 0 to 255.
        :param dimensionality: whether to do 2D or 3D connected components
                               default is 2d
        :param fg_is_higher: True if foreground is above threshold, False if
                             below
        '''
        #
        # Get the load plans for the inputs
        #
        prob_load_plan_path, plp = self.loading_plan(volume, prob_dataset_name)
        if mask_dataset_name is not None:
            mask_load_plan_path, mlp = \
                self.loading_plan(volume, mask_dataset_name)
        else:
            mask_load_plan_path = EMPTY_LOCATION
            mlp = self.__Null()
        #
        # Get the storage plan for the segmentation
        #
        storage_plan, sp = self.storage_plan(volume, seg_dataset_name)
        
        if dimensionality == Dimensionality.D2:
            task_class = SegmentCC2DTask
        else:
            task_class = SegmentCC3DTask
        task = sp ( plp ( mlp ( task_class(
            prob_loading_plan_path=prob_load_plan_path,
            mask_loading_plan_path=mask_load_plan_path,
            storage_plan=storage_plan,
            threshold=threshold,
            fg_is_higher=fg_is_higher))))
        return task

    
    def gen_unsegmentation_task(self, volume, 
                                input_dataset_name, output_dataset_name,
                                use_min_contact, contact_threshold,
                                src_task=None):
        '''Generate a 3d to 2d unsegmentation task
        
        Convert a 3d segmentation into a stack of 2d segmentations. Connected
        components is performed on the 3d volume where the edges are adjacent
        pixels in the 2d planes with the same label.
        
        :param volume:  the volume to be segmented
        :param input_dataset_name: the dataset name of the input segmentation,
        e.g. "segmentation"
        :param output_dataset_name: dataset name of the output segmentation,
        e.g. "resegmentation"
        :param use_min_contact: only break objects if they have less than
        a certain amount of contact area between planes
        :param contact_threshold: minimum area to keep an object together
        :param src_task: the task that produced the input segmentation. Default
        is to take segmentation from whatever overlaps
        '''
        #
        # Get the loading plan for the input segmentation
        #
        loading_plan_path, lp = self.loading_plan(
            volume, input_dataset_name, src_task)
        #
        # Get the storage plan for the output segmentation
        #
        storage_plan = self.storage_plan(volume, output_dataset_name)
        
        return sp ( lp ( UnsegmentTask(
            volume=volume, 
            input_loading_plan_path=loading_plan_path,
            storage_plan=storage_plan,
            use_min_contact=use_min_contact,
            contact_threshold=contact_threshold)))

    def gen_filter_task(self, volume, input_dataset_name, output_dataset_name,
                        min_area, src_task=None):
        '''Generate a task that filters small objects
        
        :param volume: the volume of the segmentation
        :param input_dataset_name: the dataset name of the input segmentation
        :param output_dataset_name: the dataset name of the output segmentation
        :param min_area: the minimum allowable area for a segment
        :param src_task: the task supplying the input segmentation
        '''
        loading_plan_path, lp = self.loading_plan(
            volume, input_dataset_name, src_task)
        storage_plan, sp = self.storage_plan(volume, output_dataset_name)
        return sp ( lp ( FilterSegmentationTask(
            volume=volume,
            input_loading_plan_path=loading_plan_path,
            storage_plan=storage_plan,
            min_area=min_area)))
    
    def gen_skeletonize_task(
        self, volume, segmentation_dataset_name, skeleton_location,
        xy_nm, z_nm, decimation_factor=0, src_task=None, 
        connectivity_graph=EMPTY_LOCATION):
        '''Generate a skeletonize task
        
        The skeletonize task takes a segmentation and produces .swc files
        (see http://research.mssm.edu/cnic/swc.html).
        
        :param volume: the volume in global coordinates of the segmentation
        :param segmentation_dataset_name: the dataset name of the segmentation
        to skeletonize
        :param skeleton_location: the name of the directory that will hold
            the .swc files.
        :param xy_nm: the size of a voxel in the x and y directions
        :param z_nm: the size of a voxel in the z direction
        :param decimation_factor: remove a skeleton leaf if it is less than
            this factor of its parent's volume.
        :param src_task: the task that produced the segmentation. Default
        is don't care
        :param connectivity_graph: the connectivity graph for making a global
        segmentation
        '''
        segmentation_loading_plan_path, lp = self.loading_plan(
            volume, segmentation_dataset_name, src_task)
        return lp ( SkeletonizeTask(
            segmentation_loading_plan_path=segmentation_loading_plan_path,
            skeleton_location=skeleton_location,
            xy_nm=xy_nm, z_nm=z_nm, 
            decimation_factor=decimation_factor,
            connectivity_graph=connectivity_graph))
    
    def gen_find_synapses_task(
        self, volume, synapse_prob_dataset_name, output_dataset_name,
        erosion_xy, erosion_z, sigma_xy, sigma_z, threshold,
        min_size_2d, max_size_2d, min_size_3d, min_slice,
        neuron_segmentation_dataset_name=None,
        neuron_src_task=None):
        '''Generate a task to segment synapses
        
        :param volume: the volume to segment
        :param synapse_prob_dataset: the dataset name of the synapse
        probabilities, e.g. "synapse"
        :param neuron_segmentation_dataset: the dataset name of the
        segmentation of the neurons. By default, do not use the neuron
        dataset.
        :param output_dataset_name: the dataset name for the synapse
        segmentation, e.g. "synapse-segmentation"
        :param erosion_xy: how much to erode neurons in the x/y direction
        :param erosion_z: how much to erode neurons in the z direction
        :param sigma_xy: The sigma for the smoothing gaussian in the x and y
                         directions
        :param sigma_z: The sigma for the smoothing Gaussian in the z direction
        :param threshold: The probability threshold above which, a voxel is
                          deemed to be part of a synapse.
        :param min_size_2d: discard any 2d segments with area less than this.
        :param max_size_2d: discard any 2d segments with area greater than this.
        :param min_size_3d: discard any 3d segments with area less than this.
        :param min_slice: discard any 3d segments whose z-extent is lt this.
        '''
        #
        # Get the loading plans
        #
        synprob_loading_plan_path, slp = self.loading_plan(
            volume, synapse_prob_dataset_name)
        if neuron_segmentation_dataset_name is None:
            neuron_loading_plan_path = EMPTY_LOCATION
            nlp = self.__Null()
            erode_with_neurons=False
        else:
            neuron_loading_plan_path, nlp = self.loading_plan(
                volume, neuron_segmentation_dataset_name, neuron_src_task)
            erode_with_neurons = True
        #
        # Get the storage plan
        #
        storage_plan, sp = self.storage_plan(volume, output_dataset_name)
        return slp ( nlp ( sp ( FindSynapsesTask(
            synapse_map_loading_plan_path=synprob_loading_plan_path,
            neuron_segmentation_loading_plan_path=neuron_loading_plan_path,
            storage_plan = storage_plan,
            erosion_xy=erosion_xy,
            erosion_z= erosion_z,
            sigma_xy=sigma_xy,
            sigma_z=sigma_z,
            threshold=threshold,
            min_size_2d=min_size_2d,
            max_size_2d=max_size_2d,
            min_size_3d=min_size_3d,
            min_slice=min_slice,
            erode_with_neurons=erode_with_neurons))))
    
    def gen_find_synapses_tr_task(
        self, volume, transmitter_dataset_name, receptor_dataset_name,
        output_dataset_name, 
        erosion_xy, erosion_z, sigma_xy, sigma_z, threshold,
        min_size_2d, max_size_2d, min_size_3d, min_slice,
        neuron_dataset_name = None,
        neuron_src_task=None):
        '''Generate a task to segment synapses w/a transmitter and receptor map
        
        :param volume: the volume to segment
        :param transmitter_dataset_name: the name of the transmitter
        probability map dataset
        :param receptor_location: the receptor probability map dataset name
        :param neuron_dataset_name: the name of the neuron segmentation dataset
        :param output_dataset_name: the name of the synapse segmentation dataset
        :param erosion_xy: how much to erode neurons in the x/y direction
        :param erosion_z: how much to erode neurons in the z direction
        :param sigma_xy: The sigma for the smoothing gaussian in the x and y
                         directions
        :param sigma_z: The sigma for the smoothing Gaussian in the z direction
        :param threshold: The probability threshold above which, a voxel is
                          deemed to be part of a synapse.
        :param min_size_2d: discard any 2d segments with area less than this.
        :param max_size_2d: discard any 2d segments with area greater than this.
        :param min_size_3d: discard any 3d segments with area less than this.
        :param min_slice: discard any 3d segments whose z-extent is lt this.
        :param neuron_src_task: the task that produced the neuron segmentation
        '''
        #
        # Get the loading plans
        #
        transmitter_loading_plan_path, tlp = self.loading_plan(
            volume, transmitter_dataset_name)
        receptor_loading_plan_path, rlp = self.loading_plan(
            volume, receptor_dataset_name)
        if neuron_dataset_name is None:
            neuron_loading_plan_path = EMPTY_LOCATION
            nlp = self.__Null()
            erode_with_neurons=False
        else:
            neuron_loading_plan_path, nlp = self.loading_plan(
                volume, neuron_dataset_name, neuron_src_task)
            erode_with_neurons = True
        #
        # Get the storage plan
        #
        storage_plan, sp = self.storage_plan(volume, output_dataset_name)
        return tlp ( rlp ( nlp ( sp ( FindSynapsesTask(
            transmitter_map_loading_plan_path=transmitter_loading_plan_path,
            receptor_map_loading_plan_path=receptor_loading_plan_path,
            neuron_segmentation_loading_plan_path=neuron_loading_plan_path,
            storage_plan = storage_plan,
            erosion_xy=erosion_xy,
            erosion_z= erosion_z,
            sigma_xy=sigma_xy,
            sigma_z=sigma_z,
            threshold=threshold,
            min_size_2d=min_size_2d,
            max_size_2d=max_size_2d,
            min_size_3d=min_size_3d,
            min_slice=min_slice,
            wants_dual_probability_maps=True,
            erode_with_neurons=erode_with_neurons)))))
    
    def gen_connect_synapses_task(
        self, volume, synapse_dataset_name, neuron_dataset_name, 
        output_location,
        xy_dilation, z_dilation, min_contact, 
        synapse_src_task=None, neuron_src_task=None,
        transmitter_dataset_name=None, 
        receptor_dataset_name=None):
        '''Generate a task to connect synapses to neurons
        
        :param volume: the volume containing the synapses and neurons
        :param synapse_dataset_name: the name of the synapse segmentation
                                 dataset
        :param neuron_dataset_name: the name of the neuron segmentation dataset
        :param output_location: the location for the json file containing
                                the connections
        :param xy_dilation: how much to dilate the synapses in the X and Y
                            directions before overlapping with neurons
        :param z_dilation: how much to dilate in the Z direction
        :param min_contact: do not connect if fewer than this many overlapping
                            voxels
        :param synapse_src_task: the task providing the synapse segmentation
        Default is take segmentation from any task.
        :param neuron_src_task: the task providing the neuron segmentation
        :param transmitter_dataset_name: the name of the pre-synaptic
        probability map (for determining synapse polarity). Default is not
        to assess polarity.
        :param receptor_dataset_name: the name of the post-synaptic probability
        map. Default is not to assess polarity .
        
        The structure of the output is a dictionary of lists. The lists
        are columns of labels and the rows are two neuron labels that
        match one segment label.
        
        The dictionary keys are "neuron_1", "neuron_2", "synapse"
        '''
        synapse_load_plan, slp = self.loading_plan(
            volume, synapse_dataset_name, synapse_src_task)
        neuron_load_plan, nlp = self.loading_plan(
            volume, neuron_dataset_name, neuron_src_task)
        additional_lps = []
        if transmitter_dataset_name is not None:
            transmitter_loading_plan_path, lp = self.loading_plan(
                volume, transmitter_dataset_name)
            additional_lps.append(lp)
            receptor_loading_plan_path, lp = self.loading_plan(
                volume, receptor_dataset_name)
            additional_lps.append(lp)
        else:
            transmitter_loading_plan_path = EMPTY_LOCATION
            receptor_loading_plan_path = EMPTY_LOCATION
        task = slp ( nlp ( ConnectSynapsesTask(
            synapse_seg_load_plan_path=synapse_load_plan,
            neuron_seg_load_plan_path=neuron_load_plan,
            transmitter_probability_map_load_plan_path=
            transmitter_loading_plan_path,
            receptor_probability_map_load_plan_path=
            receptor_loading_plan_path,
            output_location=output_location,
            xy_dilation=xy_dilation,
            z_dilation=z_dilation,
            min_contact=min_contact)))
        for additional_lp in  additional_lps:
            task = additional_lp(task)
        return task
    
    def gen_aggregate_connect_synapses_task(
        self, synapse_connection_locations, connectivity_graph_location, 
        output_location):
        '''Generate a task to combine all synapse connection files in a run
        
        :param synapse_connection_locations: a list of the locations of
             the synapse connection files generated by ConnectedComponentsTask
        :param connectivity_graph_location: the location of the connectivity
        graph file output by AllConnectedComponents
        :param output_location: the location for the aggregate file.
        '''
        return AggregateSynapseConnectionsTask(
            synapse_connection_locations=synapse_connection_locations,
            connectivity_graph_location=connectivity_graph_location,
            output_location=output_location)
    
    def gen_match_neurons_task(
        self, volume, gt_dataset_name, detected_dataset_name, 
        output_location, detected_src_task=None):
        '''Match detected neurons to ground truth based on maximum overlap

        :param volume: the volume being analyzed
        :param gt_dataset_name: the name of the dataset providing ground-truth
        segmentation of neurons
        :param detected_dataset_name: the name of the dataset providing
        the pipeline's segmentation
        :param output_location: the location on disk for the .json file
                                that gives the gt neuron that matches
                                the detected
        :param detected_src_task: the task that produced the segmentation.
        Default is take data from whatever overlaps
        '''
        gt_loading_plan, glp = self.loading_plan(volume, gt_dataset_name)
        detected_loading_plan, dlp = self.loading_plan(
            volume, detected_dataset_name, detected_src_task)
        return glp ( dlp ( MatchNeuronsTask(
            gt_load_plan_path=gt_loading_plan,
            detected_load_plan_path=detected_loading_plan,
            output_location=output_location)))
    
    def gen_match_synapses_task(
        self, volume, gt_dataset_name, detected_dataset_name, 
        output_location,
        method, mask_dataset_name=None, detected_src_task=None):
        '''Generate a task to match ground truth synapses to detected
        
        :param volume: The volume to analyze
        :param gt_dataset_name: The name of the ground-truth neurons dataset
        :param detected_dataset_name: The name of the detected neuron dataset
        :param mask_dataset_name: The name of the ground-truth mask dataset
        :param output_location: where to store the .json file with the
            synapse-synapse correlates
        :param method: one of the MatchMethod enums - either "overlap" to
        match detected and gt synapses by maximum overlap or "distance"
        to match them by closest distance.
        '''
        gt_loading_plan, glp = self.loading_plan(volume, gt_dataset_name)
        detected_loading_plan, dlp = self.loading_plan(
            volume, detected_dataset_name, detected_src_task)
        if mask_dataset_name is None:
            mask_loading_plan = EMPTY_LOCATION
            mlp = self.__Null()
        else:
            mask_loading_plan, mlp = self.loading_plan(
                volume, mask_dataset_name)
        return glp ( dlp ( mlp ( MatchSynapsesTask(
            gt_loading_plan_path=gt_loading_plan,
            detected_loading_plan_path=detected_loading_plan,
            mask_loading_plan_path=mask_loading_plan,
            output_location=output_location,
            match_method=method))))
    
    def gen_synapse_statistics_task(
        self, synapse_matches, detected_synapse_connections, neuron_map,
        gt_neuron_maps, gt_synapse_connections, output_location):
        '''Calculate precision/recall on synapse-synapse connections
        
        :param synapse_matches: .json files containing gt - detected matches
        :param detected_synapse_connections: sequence of .json files
             containing connections between detected synapses and neurons
        :param neuron_map: .json file containing output of 
            AllConnectedComponents giving the mapping of local neuron label
            to global neuron label
        :param gt_neuron_maps: .json files mstching the local labels of
            detected neurons to those of ground-truth neurons
        :param gt_synapse_connections: .json files containing connections
        between gt synapses and neurons
        '''
        return SynapseStatisticsTask(
            synapse_matches=synapse_matches,
            detected_synapse_connections=detected_synapse_connections,
            neuron_map=neuron_map,
            gt_neuron_maps=gt_neuron_maps,
            gt_synapse_connections=gt_synapse_connections,
            output_location=output_location)
        
    def __get_neuroproof_config(self, program):
        '''Return the location of the given program and its LD_LIBRARY_PATH
        
        :param program: the name of the neuroproof program to be run, e.g.
                        "neuroproof_graph_predict"
        '''
        try:
            config = rh_config.config["neuroproof"]
            neuroproof = config[program]
        except KeyError:
            raise ValueError(
                "The .rh_config.yaml file is missing configuration information "
                "for Neuroproof. See README.md for details on how to build and "
                "configure Neuroproof.")
        ld_library_path = os.pathsep.join(config.get("ld_library_path", []))
        return neuroproof, ld_library_path
        
    def gen_neuroproof_task(
        self, volume, prob_dataset_name, 
        additional_dataset_names,
        input_seg_dataset_name, 
        output_dataset_name,
        classifier_filename,
        neuroproof_version,
        input_seg_src_task=None):
        '''Run Neuroproof on an oversegmented volume
        
        :param volume: the volume being Neuroproofed
        :param prob_dataset_name: the name of the probability dataset e.g.
        "membrane"
        :param additional_dataset_names: the names of any additional
        probability maps.
        :param input_seg_dataset_name: the name of the input segmentation
        e.g. "segmentation"
        :param output_seg_dataset_name: the name of the neuroproofed
        segmentation, e.g. "neuroproof"
        :param classifier_filename: the classifier trained to assess merge/no
        merge decisions.
        :param np_version: the version of the Neuroproof binary - one
        of the NeuroproofVersion enums.
        :param input_seg_src_task: the source of the input segmentation in
        order to pick the output of a particular block.
        '''
        prob_loading_plan_path, plp = self.loading_plan(
            volume, prob_dataset_name)
        additional_loading_plan_paths = []
        additional_lps = []
        for dataset_name in additional_dataset_names:
            loading_plan_path, lp = self.loading_plan(
                volume, dataset_name)
            additional_loading_plan_paths.append(loading_plan_path)
            additional_lps.append(lp)
        input_seg_loading_plan_path, slp = self.loading_plan(
            volume, input_seg_dataset_name, input_seg_src_task)
        storage_plan, sp = self.storage_plan(volume, output_dataset_name)
        neuroproof, ld_library_path = \
            self.__get_neuroproof_config("neuroproof_graph_predict")
        task = plp ( slp ( sp ( NeuroproofTask(
            prob_loading_plan_path=prob_loading_plan_path,
            additional_loading_plan_paths=additional_loading_plan_paths,
            input_seg_loading_plan_path=input_seg_loading_plan_path,
            storage_plan=storage_plan,
            neuroproof_version=neuroproof_version,
            neuroproof=neuroproof,
            neuroproof_ld_library_path=ld_library_path,
            classifier_filename=classifier_filename))))
        for lp in additional_lps:
            task = lp(task)
        return task

    def gen_stitched_neuroproof_task(
        self, volume, prob_dataset_name, 
        additional_dataset_names,
        input_seg_dataset_name, 
        output_dataset_name,
        classifier_filename,
        neuroproof_version,
        dilate_xy=7,
        dilate_z=5,
        input_seg_src_task=None):
        '''Run Neuroproof on an oversegmented volume
        
        :param volume: the volume being Neuroproofed
        :param prob_dataset_name: the name of the probability dataset e.g.
        "membrane"
        :param additional_dataset_names: the names of any additional
        probability maps.
        :param input_seg_dataset_name: the name of the input segmentation
        e.g. "segmentation"
        :param output_seg_dataset_name: the name of the neuroproofed
        segmentation, e.g. "neuroproof"
        :param classifier_filename: the classifier trained to assess merge/no
        merge decisions.
        :param np_version: the version of the Neuroproof binary - one
        of the NeuroproofVersion enums.
        :param dilate_xy: amount to dilate x and y membrane
        :param dilate_z: amount to dilate z membrane
        :param input_seg_src_task: the source of the input segmentation in
        order to pick the output of a particular block.
        '''
        prob_loading_plan_path, plp = self.loading_plan(
            volume, prob_dataset_name)
        additional_loading_plan_paths = []
        additional_lps = []
        for dataset_name in additional_dataset_names:
            loading_plan_path, lp = self.loading_plan(
                volume, dataset_name)
            additional_loading_plan_paths.append(loading_plan_path)
            additional_lps.append(lp)
        input_seg_loading_plan_path, slp = self.loading_plan(
            volume, input_seg_dataset_name, input_seg_src_task)
        storage_plan, sp = self.storage_plan(volume, output_dataset_name)
        neuroproof, ld_library_path = \
            self.__get_neuroproof_config("neuroproof_graph_predict")
        task = plp ( slp ( sp ( NeuroproofStitchTask(
            prob_loading_plan_path=prob_loading_plan_path,
            additional_loading_plan_paths=additional_loading_plan_paths,
            input_seg_loading_plan_path=input_seg_loading_plan_path,
            storage_plan=storage_plan,
            neuroproof_version=neuroproof_version,
            neuroproof=neuroproof,
            neuroproof_ld_library_path=ld_library_path,
            classifier_filename=classifier_filename,
            dilation_xy=dilate_xy,
            dilation_z=dilate_z))))
        for lp in additional_lps:
            task = lp(task)
        return task

    def gen_neuroproof_learn_task(self,
                                  volume,
                                  prob_dataset_name,
                                  seg_dataset_name,
                                  gt_dataset_name,
                                  output_location,
                                  strategy=StrategyEnum.all,
                                  num_iterations=1,
                                  prune_feature=True,
                                  use_mito=False,
                                  neuroproof_version=NeuroproofVersion.MIT):
        '''Generate a task to learn a Neuroproof classifier
        
        :param volume: the volume for the probability, segmentation and
                       ground-truth data.
        :param prob_dataset_name: the name of the probability dataset, e.g.
        "membrane"
        :param seg_dataset_name: the name of the segmentation dataset
                       produced by the pipeline
        :param gt_dataset_name: the name of the ground truth segmentation
        :param output_location: the location for the classifier file.
        :param strategy: the strategy to use for classification.
        :param num_iterations: the number of times to refine the classifier
        :param prune_feature: True to prune features with low predictive
               values
        :param use_mito: True to use a mitochondrial channel.
        :param neuroproof_version: choose from among the Neuroproof binary
        command-line formats - MIT, FLY_EM or MINIMAL
        '''
        neuroproof, ld_library_path = self.__get_neuroproof_config(
            "neuroproof_graph_learn")
        prob_loading_plan, plp = self.loading_plan(volume, prob_dataset_name)
        seg_loading_plan, slp = self.loading_plan(volume, seg_dataset_name)
        gt_loading_plan, glp = self.loading_plan(volume, gt_dataset_name)
        
        return plp ( slp ( glp ( NeuroproofLearnTask(
            prob_loading_plan_path=prob_loading_plan,
            seg_loading_plan_path=seg_loading_plan,
            gt_loading_plan_path=gt_loading_plan,
            output_location=output_location,
            neuroproof=neuroproof,
            neuroproof_ld_library_path=ld_library_path,
            strategy=strategy,
            num_iterations=num_iterations,
            prune_feature=prune_feature,
            use_mito=use_mito,
            neuroproof_version=neuroproof_version))))
    
    def gen_mask_border_task(
        self, volume, prob_dataset_name, mask_dataset_name, threshold=250):
        '''Generate the outer and border masks for a volume
        
        :param volume: the volume being masked
        :param prob_dataset_name: the name of the membrane probability dataset,
        e.g. "membrane"
        :param mask_dataset_name: the name for the mask dataset, e.g. "mask"
        :param threshold: Mask out voxels with membrane probabilities at this
        threshold or higher.
        '''
        prob_loading_plan_path, lp = self.loading_plan(
            volume, prob_dataset_name)
        storage_plan, sp = self.storage_plan(volume, mask_dataset_name)
        return lp(sp(MaskBorderTask(
            prob_loading_plan_path=prob_loading_plan_path,
            storage_plan=storage_plan,
            threshold=threshold)))
    
    def gen_connected_components_task(
        self, dataset_name, volume1, src_task1, volume2, src_task2, 
        overlap_volume, output_location):
        '''Find the connected components between two segmentation volumes
        
        :param dataset_name: the name of the segmentation dataset, 
        e.g. "neuroproof"
        :param volume1: the volume of the first segmentation
        :param src_task1: the task that produced the first volume
        :param volume2: the volume of the second segmentation
        :param src_task2: the task that produced the second volume
        :param overlap_volume: the volume to be scanned for connected components
        :param output_location: where to write the data file
        '''
        #
        # Lots of loading plans here: a cutout and complete one for each
        # segmentation.
        #
        cutout1, c1lp = self.loading_plan(
            overlap_volume, dataset_name, src_task1)
        full1, f1lp = self.loading_plan(volume1, dataset_name, src_task1)
        cutout2, c2lp = self.loading_plan(
            overlap_volume, dataset_name, src_task2)
        full2, f2lp = self.loading_plan(volume2, dataset_name, src_task2)
        
        return c1lp( f1lp( c2lp( f2lp( ConnectedComponentsTask(
            volume1=volume1,
            cutout_loading_plan1_path=cutout1,
            segmentation_loading_plan1_path=full1,
            volume2=volume2,
            cutout_loading_plan2_path=cutout2,
            segmentation_loading_plan2_path=full2,
            output_location=output_location)))))
    
    def gen_abutting_connected_components_task(
        self, dataset_name, 
        volume1, src_task1, sliver_volume1,
        volume2, src_task2, sliver_volume2, 
        overlap_volume, neuroproof_task, 
        neuroproof_dataset_name, output_location):
        '''Create a connected components task using the abutting method
        
        :param dataset_name: the name of the segmentation dataset, 
        e.g. "neuroproof"
        :param volume1: the volume of the first segmentation
        :param src_task1: the task that produced the first volume
        :param sliver_volume1: the volume to take out of the first task's
        segmentation when comparing
        :param volume2: the volume of the second segmentation
        :param src_task2: the task that produced the second volume
        :param sliver_volume2: the volume to take out of the second task's
        segmentation when comparing
        :param overlap_volume: the volume to be scanned for connected components
        :param neuroproof_task: the task that performed the neuroproofing of
        the chimera segmentation of the two blocks
        :param neuroproof_dataset_name: the name of the dataset used in
        the chimeric neuroproofing
        :param output_location: where to write the data file
        '''
        #
        # Lots of loading plans here: a cutout and complete one for each
        # segmentation.
        #
        cutout1, c1lp = self.loading_plan(
            sliver_volume1, dataset_name, src_task1)
        full1, f1lp = self.loading_plan(volume1, dataset_name, src_task1)
        cutout2, c2lp = self.loading_plan(
            sliver_volume2, dataset_name, src_task2)
        full2, f2lp = self.loading_plan(volume2, dataset_name, src_task2)
        neuroproof_loading_plan, nplp = self.loading_plan(
            overlap_volume, neuroproof_dataset_name, neuroproof_task)
        
        return nplp(c1lp( f1lp( c2lp( f2lp( ConnectedComponentsTask(
            volume1=volume1,
            cutout_loading_plan1_path=cutout1,
            segmentation_loading_plan1_path=full1,
            volume2=volume2,
            cutout_loading_plan2_path=cutout2,
            segmentation_loading_plan2_path=full2,
            neuroproof_segmentation = neuroproof_loading_plan,
            output_location=output_location))))))
    
    def gen_join_connected_components_task(
        self, connectivity_graph1, index1,
        connectivity_graph2, index2,
        output_location):
        '''Join the outputs of two connected components tasks
        
        :param connectivity_graph1: the output of a ConnectedComponentsTask
        :param index1: the index (0, or 1) of the component to keep from
        the first graph. The other becomes the component to join to the
        second graph.
        :param connectivity_graph2: the output of a ConnectedComponentsTask
        :param index2: the index (0, or 1) of the component to keep from
        the second graph. The other becomes the component to join to the
        first graph.
        :param output_location: where to store the resulting file
        '''
        return JoinConnectedComponentsTask(
            connectivity_graph1=connectivity_graph1,
            index1=index1,
            connectivity_graph2=connectivity_graph2,
            index2=index2,
            output_location=output_location)

    def gen_all_connected_components_task(
        self, input_locations, output_location, additional_loading_plans = []):
        '''Construct the global mapping for local segmentations
        
        Given the concordances from the ConnectedComponentsTasks of an
        analysis volume, construct a mapping from each volume's
        segmentation labels to a global mapping that unites adjacent
        components from adjacent volumes.
        
        :param input_locations: a sequence of pathnames
        of the outputs generated by the ConnectedComponentsTasks
        :param output_location: the path name for the global mapping file
        :param additional_loading_plans: a list of two-tuples of volume
        and loading plan for additional loading plans, e.g. for
        connecting cube edges.
        '''
        return AllConnectedComponentsTask(
            input_locations=input_locations,
            output_location=output_location,
            additional_loading_plans=additional_loading_plans)
    
    def gen_fake_all_connected_components_task(
        self, volume, dataset_name, output_location, src_task=None):
        '''Generate a connectivity graph file for a single location'''
        loading_plan, lp = self.loading_plan(volume, dataset_name, src_task)
        return lp(FakeAllConnectedComponentsTask(
            volume=volume,
            loading_plan=loading_plan,
            output_location=output_location))
    
    def gen_volume_relabeling_task(
        self, dataset_name, input_volumes, relabeling_location, 
        output_volume, output_dataset_name):
        '''Relabel a segmentation using global labels
        
        Use the output of the AllConnectedComponents task to map
        input segmentations to a single output segmentation using
        globally-valid labels.
        
        :param dataset_name: the name of the segmentation dataset
        :param input_volumes: a sequence of dictionaries having keys of
        "volume" and "task". Each "volume" is a
        :py:class:`ariadne_microns_pipeline.parameters.Volume` describing
        the volume of the input dataset. Each "task" is the
        :py:class:`luigi.Task` that produced the volume.
        :param relabeling_location: the location of the file produced
        by the AllConnectedComponentsTask
        :param output_volume: the volume to write
        :param output_dataset_name: the name of the output volume dataset
        '''
        loading_plans = []
        lp_objects = []
        for d in input_volumes:
            volume = d["volume"]
            task = d["task"]
            loading_plan, lp = self.loading_plan(volume, dataset_name, task)
            loading_plans.append(loading_plan)
            lp_objects.append(lp)
        storage_plan, sp = self.storage_plan(output_volume, output_dataset_name)
        task = sp ( VolumeRelabelingTask(
            input_volumes=loading_plans,
            relabeling_location=relabeling_location,
            storage_plan=storage_plan))
        for lp in lp_objects:
            lp(task)
        return task
    
    def gen_copy_storage_plan_task(
        self, storage_plan_path, volume, dataset_name):
        '''Copy a storage plan from some other pipeline
        
        :param storage_plan_path: original storage plan
        :param volume: the volume of the target storage plan
        :param dataset_name: the name of the dataset.
        '''
        new_storage_plan_path, sp = self.storage_plan(volume, dataset_name)
        task = sp(CopyStoragePlanTask(
            storage_plan=new_storage_plan_path,
            src_storage_plan=storage_plan_path))
        return task
        
    def gen_copy_loading_plan_task(
        self, loading_plan_path, volume, dataset_name, offset = 0):
        '''Copy a loading plan from some other pipeline
        
        :param loading_plan_path: original storage plan
        :param volume: the volume of the target storage plan
        :param dataset_name: the name of the dataset.
        :param offset: the offset to add to each nonzero voxel value
        '''
        new_storage_plan_path, sp = self.storage_plan(volume, dataset_name)
        task = sp(CopyLoadingPlanTask(
            storage_plan=new_storage_plan_path,
            src_loading_plan=loading_plan_path,
            offset=offset))
        return task
    
    def gen_aggregate_loading_plan_tasks(
        self, volume, input_datasets, output_dataset, operation, 
        src_tasks=None):
        '''Generate an AggregatLoadingPlansTask
        
        :param volume: the volume to process
        :param input_datasets: the dataset names of the input datasets
        :param output_dataset: the dataset name of the output dataset
        :param operation: the name of the operation, from
                          copytasks.AggregateOperation
        :param src_tasks: the source tasks for each of the datasets, defaults
        to any task.
        '''
        loading_plans, lps = [], []
        if src_tasks is None:
            src_tasks = [None] * len(input_datasets)
            for dataset_name, src_task in zip(input_datasets, src_tasks):
                loading_plan, lp = self.loading_plan(
                    volume, dataset_name, src_task)
                loading_plans.append(loading_plan)
                lps.append(lp)
        storage_plan, sp = self.storage_plan(volume, output_dataset)
        task = AggregateLoadingPlansTask(
           loading_plan_paths=loading_plans,
           operation=operation,
           storage_plan = storage_plan)
        for lp in lps:
            task = lp(task)
        return sp(task)
        
    def gen_storage_plan_relabeling_task(
        self, connectivity_graph_path, volume, src_loading_plan,
        dataset_name=None):
        '''Generate a storage plan relabeling task
        
        This task takes a loading plan from a previous pipeline and
        generates a task that relabels the data and then stores it
        in a new storage plan.
        
        :param connectivity_graph_path: path to connectivity graph for
        relabeling
        :param volume: volume for storage plan
        :param src_loading_plan: loading plan from other pipeline
        :param dataset_name: dataset name for the storage plan. Default is
        the same as that of the loading plan
        '''
        if dataset_name is None:
            loading_plan = DestVolumeReader(src_loading_plan)
            dataset_name = loading_plan.dataset_name
        new_storage_plan, sp = self.storage_plan(volume, dataset_name)
        task = sp(StoragePlanRelabelingTask(
            connectivity_graph_path=connectivity_graph_path,
            storage_plan=new_storage_plan,
            src_loading_plan_path=src_loading_plan))
        return task
    
    def gen_synapse_relabeling_task(
        self, synapse_connections_path, volume, loading_plan, 
        dataset_name=None):
        '''Generate a synapse relabeling task
        
        This task takes a loading plan for a synapse segmentation and
        relabels it based on the locations of synapses in the synapse
        connectivity file.
        
        :param synapse_connections_path: the path to the
        synapse-connections.json file that has the locations of every
        synapse in the experiment.
        :param volume: the volume encompassing the synapse segmentation
        :param loading_plan: the name of the loading plan file for the
        synapse segmentation
        :param dataset_name: the name of the output dataset
        '''
        if dataset_name is None:
            loading_plan = DestVolumeReader(src_loading_plan)
            dataset_name = loading_plan.dataset_name
        new_storage_plan, sp = self.storage_plan(volume, dataset_name)
        task = sp(SynapseRelabelingTask(
            synapse_connections_path=synapse_connections_path,
            storage_plan=new_storage_plan,
            loading_plan=loading_plan))
        return task
    
    def gen_chimeric_segmentation_task(
        self, volume1, task1, volume2, task2, 
        input_dataset_name, output_dataset_name):
        '''Generate a task to take two abutting segmentations and combine.
        
        This creates a chimeric segmentation, half one segmentation, half
        the other. This can then be fed to Neuroproof which will knit the two
        together.
        
        :param volume1: the volume encompassing the first segmentation
        :param task1: the task that generated the segmentation
        :param volume2: the volume encompassing the second segmentation
        :param task2: the task that generated the segmentation
        :param input_dataset_name: the name of the dataset for the input volumes
        :param output_dataset_name: the name of the output dataset
        '''
        loading_plan_1, lp1 = self.loading_plan(
             volume1, input_dataset_name, task1)
        loading_plan_2, lp2 = self.loading_plan(
             volume2, input_dataset_name, task2)
        volume = volume1.get_union_region(volume2)
        storage_plan, sp = self.storage_plan(volume, output_dataset_name)
        task = ChimericSegmentationTask(
            loading_plan1_path=loading_plan_1,
            loading_plan2_path=loading_plan_2,
            storage_plan=storage_plan)
        return sp(lp1(lp2(task)))
    
    def gen_boss_sharding_task(self,
                               volume,
                               dataset_name,
                               output_dtype,
                               pattern,
                               done_file,
                               compression=3,
                               src_task=None):
        '''Generate a Boss sharding task for the given volume
        
        :param volume: Shard this volume into planes
        :param dataset_name: the name of the dataset to be sharded
        :param output_dtype: the Numpy dtype of the planes
        :param pattern: the file naming pattern, suitable for use in
            pattern.format(x=x, y=y, z=z)
        :param done_file: the done file that marks the task being finished
        :param compression: 0-9, the compression factor to use
        :param src_task: the source of the  data
        '''
        loading_plan_path, lp = self.loading_plan(
            volume, dataset_name, src_task)
        task = BossShardingTask(
            loading_plan_path=loading_plan_path,
            pattern=pattern,
            done_file=done_file,
            output_dtype=output_dtype,
            compression=compression)
        return lp(task)
                               
    #########################
    #
    # Helper tasks
    #
    #    These help bridge one task to another.
    #
    #########################

    def gen_segmentation_statistics_task(self,
                                         volume,
                                         block_volume,
                                         gt_seg_dataset_name,
                                         pred_seg_dataset_name,
                                         connectivity,
                                         output_location,
                                         pred_src_task=None):
        '''Collect statistics on the accuracy of a prediction
        
        This task does a statistical analysis of the accuracy of a prediction,
        comparing it against the ground truth. The data are saved as a
        JSON dictionary.
        
        :param volume: The volume to have statistics performed
        :param block_volume: the volume for table lookup into the connectivity
                             graph
        :param gt_seg_dataset_name: the location of the ground truth volume
        :param pred_seg_dataset_name: the location of the classifier prediction
        :param connectivity: the location of the connectivity-graph.json file
                             generated by the AllConnectedComponentsTask.
                             "/dev/null" if you don't have one.
        :param output_location: where to put the JSON file that contains the
        :param pred_src_task: get the prediction data produced by this task.
        Default is to get it from whatever task.
        statistics.
        '''
        gt_load_plan, glp = self.loading_plan(volume, gt_seg_dataset_name)
        pred_load_plan, plp = self.loading_plan(
            volume, pred_seg_dataset_name, pred_src_task)
        return glp ( plp ( SegmentationStatisticsTask(
            ground_truth_loading_plan_path=gt_load_plan,
            test_loading_plan_path=pred_load_plan,
            block_volume=block_volume,
            connectivity=connectivity,
            output_path=output_location)))
    
    def gen_segmentation_report_task(self,
                                     statistics_locations,
                                     pdf_location):
        '''Generate a statistics report on the segmentation
        
        :param statistics_locations: the paths to each of the JSON output
               files from the SegmentationStatisticsTasks
        :param pdf_location: where to write the matplotlib report
        '''
        return SegmentationReportTask(
            statistics_locations=statistics_locations,
            pdf_location=pdf_location)
    
    def gen_json_to_csv_task(self,
                             json_paths,
                             output_path,
                             excluded_keys=[]):
        '''Collect a number of JSON dictionaries into one .csv file
        
        Each JSON dictionary is a row in the CSV files and the columns are
        the keys in the dictionary, sorted alphabetically.
        
        :param json_paths: a sequence of pathnames to the JSON input files
        :param output_path: the location for the .csv file
        :param excluded_keys: keys to exclude from the CSV
        '''
        return JSONToCSVTask(json_paths=json_paths,
                             output_path=output_path,
                             excluded_keys=excluded_keys)
    
    def gen_stitch_segmentation_task(self,
                                     connected_components_location,
                                     output_volume,
                                     output_location,
                                     xy_chunking = 2048,
                                     z_chunking = 4,
                                     compression = Compression.GZIP):
        '''Generate a task to stitch all the segmentations together
        
        connected_components_location: the location of the connected components
                       JSON file that is an output of AllConnectedComponentsTask
        output_volume: the volume coordinates of the output volume
        output_location: the dataset location for the output HDF5 file
        xy_chunking: the chunk size in the x and y directions for the HDF5
                     dataset
        z_chunking: the chunk size in the z direction for the HDF5 dataset
        compression: one of the compression enumerations from 
                     stitch_segmentation.Compression
        '''
        return StitchSegmentationTask(
            connected_components_location=connected_components_location,
            output_volume=output_volume,
            output_location=output_location,
            xy_chunking=xy_chunking,
            z_chunking=z_chunking,
            compression=compression)

    def gen_distance_transform_task(
        self,
        volume,
        input_dataset_name,
        input_type,
        output_dataset_name):
        '''Generate a task to compute the distance transform on a volume
        
        :param volume: the volume to be analyzed
        :param input_dataset_name: the name of the input dataset
        :param input_type: an enum of DistanceTransformInputType.
            BinaryMask for a binary mask, ProbabilityMap to threshold
            a probability map to get background / foreground, Segmentation
            to get the background from the borders between segments.
        :param output_dataset_name: the name of the output's dataset
        '''
        loading_plan, lp = self.loading_plan(volume, input_dataset_name)
        storage_plan, sp = self.storage_plan(volume, output_dataset_name)
        return lp | sp | DistanceTransformTask(
            volume=volume,
            input_location=input_location,
            input_type=input_type,
            output_location=output_location)
    
    def gen_z_watershed_task(
        self,
        volume,
        x_prob_dataset_name,
        y_prob_dataset_name,
        z_prob_dataset_name,
        output_dataset_name,
        threshold=40000,
        low=1,
        high=254,
        min_size=0):
        '''Generate a Z-watershed task to segment a volume using affinities
        
        :param volume: the volume to be segmented
        :param x_prob_dataset_name: the name of the probability map dataset of
        affinities between voxels in the X direction.
        :param y_prob_dataset_name: the name of the probability map dataset of
        affinities between voxels in the Y direction.
        :param z_prob_dataset_name: the name of the probability map dataset of
        affinities between voxels in the Z direction.
        :param output_dataset_name: the name of the output dataset
        '''
        xprob_loading_plan, xlp = self.loading_plan(volume, x_prob_dataset_name)
        yprob_loading_plan, ylp = self.loading_plan(volume, y_prob_dataset_name)
        zprob_loading_plan, zlp = self.loading_plan(volume, z_prob_dataset_name)
        storage_plan, sp = self.storage_plan(volume, output_dataset_name)
        return xlp ( ylp ( zlp ( sp ( ZWatershedTask(
            volume=volume,
            x_prob_loading_plan_path=xprob_loading_plan,
            y_prob_loading_plan_path=yprob_loading_plan,
            z_prob_loading_plan_path=zprob_loading_plan,
            storage_plan=storage_plan,
            threshold=threshold,
            low=low,
            high=high,
            min_size=min_size)))))

    def gen_water_z_task(
        self,
        volume,
        x_prob_dataset_name,
        y_prob_dataset_name,
        z_prob_dataset_name,
        output_dataset_name,
        threshold=100.0,
        low_threshold=1,
        high_threshold=254):
        '''Generate a water-z task to segment a volume using affinities
        
        water-z is the Z-watershed library with agglomeration.
        
        :param volume: the volume to be segmented
        :param x_prob_dataset_name: the name of the probability map dataset of
        affinities between voxels in the X direction.
        :param y_prob_dataset_name: the name of the probability map dataset of
        affinities between voxels in the Y direction.
        :param z_prob_dataset_name: the name of the probability map dataset of
        affinities between voxels in the Z direction.
        :param output_dataset_name: the name of the output dataset
        :param threshold: the threshold for the scoring function
        :param low_threshold: the low bounds for the Z-watershed (never
        join below this threshold)
        :param high_threshold: the high bounds for the Z-watershed (always
        join above this threshold)
        '''
        xprob_loading_plan, xlp = self.loading_plan(volume, x_prob_dataset_name)
        yprob_loading_plan, ylp = self.loading_plan(volume, y_prob_dataset_name)
        zprob_loading_plan, zlp = self.loading_plan(volume, z_prob_dataset_name)
        storage_plan, sp = self.storage_plan(volume, output_dataset_name)
        return xlp ( ylp ( zlp ( sp ( WaterZTask(
            volume=volume,
            x_prob_loading_plan_path=xprob_loading_plan,
            y_prob_loading_plan_path=yprob_loading_plan,
            z_prob_loading_plan_path=zprob_loading_plan,
            storage_plan=storage_plan,
            threshold=threshold,
            low_threshold=low_threshold,
            high_threshold=high_threshold)))))