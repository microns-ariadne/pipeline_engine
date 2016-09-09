'''The task factory creates tasks for particular pipeline steps.'''

import json
import luigi
import os
import rh_config

from .download_from_butterfly import DownloadFromButterflyTask
from .block import BlockTask
from .classify import ClassifyTask
from .connected_components import AllConnectedComponentsTask
from .connected_components import ConnectedComponentsTask
from .connected_components import VolumeRelabelingTask
from .find_seeds import FindSeedsTask, Dimensionality, SeedsMethodEnum
from .json_to_csv_task import JSONToCSVTask
from .mask import MaskBorderTask
from .neuroproof import NeuroproofTask
from .nplearn import NeuroproofLearnTask, StrategyEnum
from .segment import SegmentTask, SegmentCC2DTask, UnsegmentTask
from .segmentation_statistics import \
     SegmentationStatisticsTask, SegmentationReportTask
from .skeletonize import SkeletonizeTask
from .utilities import to_hashable

class AMTaskFactory(object):
    '''Factory for creating Ariadne/Microns tasks
    
    Each method has its output target first and input targets following,
    e.g.
    
    def gen_foo_from_bar_and_baz(foo, bar, baz)
    
    The return value is the task that performs the action
    '''
   
    def gen_get_volume_task(self,
                            experiment,
                            sample,
                            dataset,
                            channel,
                            url,
                            volume,
                            location):
        '''Get a 3d volume
        
        :param experiment: the experiment done to produce the sample
        :param sample: the sample ID of the tissue that was imaged
        :param dataset: the volume that was imaged
        :param channel: the channel supplying the pixel values
        :param url: the URL of the butterfly server
        :param volume: the volume to fetch
        :type volume: :py:class:`ariadne_microns_pipeline.parameters.Volume`
        :param location: the location on disk to write the volume data
        :type location: 
            :py:class:`ariadne_microns_pipeline.parameters.DatasetLocation`
        :returns: A task that outputs a volume target.
        '''
        
        return DownloadFromButterflyTask(experiment=experiment,
                                         sample=sample,
                                         dataset=dataset,
                                         channel=channel,
                                         url=url,
                                         volume=volume,
                                         destination=location)

    def gen_classify_task(
        self, paths, datasets, pattern, img_volume, img_location,
        classifier_path):
        '''Classify a volume

        :param paths: the root paths to use for sharding
        :param datasets: a dictionary with keys of the class indexes or names
             produced by the classifier and values of the names of the
             datasets to be stored (not all datasets from the classifier need
             be stored)
        :param pattern: the pattern to use for naming files.
        :param img_volume: the image to be classified
        :param classifier_path: path to a pickled classifer
        '''
        datasets = to_hashable(datasets)
        
        return ClassifyTask(classifier_path=classifier_path,
                            volume=img_volume,
                            image_location=img_location,
                            prob_roots=paths,
                            class_names=datasets,
                            pattern=pattern)
    
    def gen_find_seeds_task(
        self, volume, prob_location, seeds_location,
        dimensionality=Dimensionality.D3, method=SeedsMethodEnum.Smoothing,
        sigma_xy=3, sigma_z=.4, threshold=1, minimum_distance_xy=5,
        minimum_distance_z=1.5, distance_threshold=5):
        '''Generate a seed finding task
        
        This task produces seeds for watershedding.
        
        :param volume: the volume in which to find the seeds
        :param prob_location: the location of the input probability dataset
        :param seeds_location: the location for the seed labels
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
        return FindSeedsTask(volume=volume,
                             prob_location=prob_location,
                             seeds_location=seeds_location,
                             dimensionality=dimensionality,
                             method=method,
                             sigma_xy=sigma_xy,
                             sigma_z=sigma_z,
                             threshold=threshold,
                             minimum_distance_xy=minimum_distance_xy,
                             minimum_distance_z=minimum_distance_z,
                             distance_threshold=distance_threshold)
    
    def gen_segmentation_task(
        self, volume, prob_location, seeds_location, mask_location,
        seg_location, sigma_xy, sigma_z, dimensionality):
        '''Generate a segmentation task

        Generate a segmentation task.  The task takes a probability map of
        the membranes and a mask of areas to exclude from segmentation. It
        smooths the membrane probabilities with an anisotropic Gaussian
        (different sigmas in XY and Z), thresholds to get
        the seeds for the watershed, then performs a 3d watershed.

        :param volume: the volume to be segmented in global coordinates
        :param prob_location: where to find the membrane probability volume
        :param mask_location: where to find the mask location
        :param seeds_location: where to find the seeds for the watershed
        :param seg_location: where to put the segmentation
        :param sigma_xy: the sigma of the smoothing gaussian in the X and Y
        directions
        :param sigma_z: the sigma of the smoothing gaussian in the Z direction
        :param dimensionality: Whether to do 2D or 3D segmentation
        '''
        return SegmentTask(volume=volume, 
                           prob_location=prob_location,
                           mask_location=mask_location,
                           seed_location=seeds_location,
                           output_location=seg_location,
                           sigma_xy=sigma_xy,
                           sigma_z=sigma_z,
                           dimensionality=dimensionality)
    
    def gen_cc_segmentation_task(
        self, volume, prob_location, mask_location, seg_location, threshold):
        '''Generate a 2d segmentation task
        
        Generate a 2D segmentation task that performs connected components
        on the individual planes.
        
        :param volume: the volume to be segmented, in global coordinates
        :param prob_location: where to find the membrane probability volume
        :param mask_location: where to find the mask location
        :param seg_location: where to put the segmentation
        :param threshold: the cutoff in the membrane probabilities between
        membrane and not-membrane, scaled from 0 to 255.
        '''
        return SegmentCC2DTask(volume=volume,
                             prob_location=prob_location,
                             mask_location=mask_location,
                             output_location=seg_location,
                             threshold=threshold)
    
    def gen_unsegmentation_task(self, volume, input_location, output_location,
                                use_min_contact, contact_threshold):
        '''Generate a 3d to 2d unsegmentation task
        
        Convert a 3d segmentation into a stack of 2d segmentations. Connected
        components is performed on the 3d volume where the edges are adjacent
        pixels in the 2d planes with the same label.
        
        :param volume:  the volume to be segmented
        :param input_location: the location of the input segmentation
        :param output_location: the location for the output segmentation
        :param use_min_contact: only break objects if they have less than
        a certain amount of contact area between planes
        :param contact_threshold: minimum area to keep an object together
        '''
        return UnsegmentTask(volume=volume, 
                             input_location=input_location,
                             output_location=output_location,
                             use_min_contact=use_min_contact,
                             contact_threshold=contact_threshold)
    
    def gen_skeletonize_task(
        self, volume, segmentation_location, skeleton_location):
        '''Generate a skeletonize task
        
        The skeletonize task takes a segmentation and produces .swc files
        (see http://research.mssm.edu/cnic/swc.html).
        
        :param volume: the volume in global coordinates of the segmentation
        :param segmentation_location: the location of the segmentation volume
        :param skeleton_location: the name of the directory that will hold
            the .swc files.
        '''
        return SkeletonizeTask(
            volume=volume,
            segmentation_location=segmentation_location,
            skeleton_location=skeleton_location)
        

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
        self, volume, prob_location, input_seg_location, output_seg_location,
        classifier_filename):
        '''Run Neuroproof on an oversegmented volume
        
        :param volume: the volume being Neuroproofed
        :param prob_location: the location of the membrane probabilities
        :param input_seg_location: the location of the oversegmentation
        :param output_seg_location: where to write the corrected segmentation
        :param classifier_filename: the classifier trained to assess merge/no
        merge decisions.
        '''
        neuroproof, ld_library_path = \
            self.__get_neuroproof_config("neuroproof_graph_predict")
        return NeuroproofTask(volume=volume,
                              prob_location=prob_location,
                              input_seg_location=input_seg_location,
                              output_seg_location=output_seg_location,
                              neuroproof=neuroproof,
                              neuroproof_ld_library_path=ld_library_path,
                              classifier_filename=classifier_filename)

    def gen_neuroproof_learn_task(self,
                                  volume,
                                  prob_location,
                                  seg_location,
                                  gt_location,
                                  output_location,
                                  strategy=StrategyEnum.all,
                                  num_iterations=1,
                                  prune_feature=True,
                                  use_mito=False):
        '''Generate a task to learn a Neuroproof classifier
        
        :param volume: the volume for the probability, segmentation and
                       ground-truth data.
        :param prob_location: the location of the probability volume
        :param seg_location: the location of the segmentation volume
                       produced by the pipeline
        :param gt_location: the location of the ground truth segmentation
        :param output_location: the location for the classifier file.
        :param strategy: the strategy to use for classification.
        :param num_iterations: the number of times to refine the classifier
        :param prune_feature: True to prune features with low predictive
               values
        :param use_mito: True to use a mitochondrial channel.
        '''
        neuroproof, ld_library_path = self.__get_neuroproof_config(
            "neuroproof_graph_learn")
        return NeuroproofLearnTask(
            volume=volume,
            prob_location=prob_location,
            seg_location=seg_location,
            gt_location=gt_location,
            output_location=output_location,
            neuroproof=neuroproof,
            neuroproof_ld_library_path=ld_library_path,
            strategy=strategy,
            num_iterations=num_iterations,
            prune_feature=prune_feature,
            use_mito=use_mito)
    
    def gen_mask_border_task(
        self, volume, prob_location, mask_location, border_width, close_width):
        '''Generate the outer and border masks for a volume
        
        :param volume: the volume being masked
        :param prob_location: the location of the membrane probabilities
        :param mask_location: where to write the masks
        :param border_width: The width of the border to consider masking
        :param close_width: the size of the square block to be used in the
             morphological closing operation.
        '''
        return MaskBorderTask(volume=volume,
                              prob_location=prob_location,
                              mask_location=mask_location,
                              border_width=border_width,
                              close_width=close_width)
    
    def gen_connected_components_task(
        self, volume1, location1, volume2, location2, overlap_volume,
        output_location):
        '''Find the connected components between two segmentation volumes
        
        :param volume1: the volume of the first segmentation
        :param location1: the location of the first segmentation's dataset
        :param volume2: the volume of the second segmentation
        :param location2: the location of the second segmentation's dataset
        :param overlap_volume: the volume to be scanned for connected components
        :param output_location: where to write the data file
        '''
        return ConnectedComponentsTask(volume1=volume1,
                                       location1=location1,
                                       volume2=volume2,
                                       location2=location2,
                                       overlap_volume=overlap_volume,
                                       output_location=output_location)
    
    def gen_all_connected_components_task(
        self, input_locations, output_location):
        '''Construct the global mapping for local segmentations
        
        Given the concordances from the ConnectedComponentsTasks of an
        analysis volume, construct a mapping from each volume's
        segmentation labels to a global mapping that unites adjacent
        components from adjacent volumes.
        
        :param input_locations: a sequence of pathnames
        of the outputs generated by the ConnectedComponentsTasks
        :param output_location: the path name for the global mapping file
        '''
        return AllConnectedComponentsTask(input_locations=input_locations,
                                          output_location=output_location)
    
    def gen_volume_relabeling_task(
        self, input_volumes, relabeling_location, 
        output_volume, output_location):
        '''Relabel a segmentation using global labels
        
        Use the output of the AllConnectedComponents task to map
        input segmentations to a single output segmentation using
        globally-valid labels.
        
        :param input_volumes: a sequence of dictionaries having keys of
        "volume" and "location". Each "volume" is a
        :py:class:`ariadne_microns_pipeline.parameters.Volume` describing
        the volume of the input dataset. Each "location" is a
        :py:class:`ariadne_microns_pipeline.parameters.DatasetLocation`
        describing the location of the dataset on disk.
        :param relabeling_location: the location of the file produced
        by the AllConnectedComponentsTask
        :param output_volume: the volume to write
        :param output_location: the location for the output volume dataset
        '''
        return VolumeRelabelingTask(input_volumes=input_volumes,
                                    relabeling_location=relabeling_location,
                                    output_volume=output_volume,
                                    output_location=output_location)
    
    #########################
    #
    # Helper tasks
    #
    #    These help bridge one task to another.
    #
    #########################

    def gen_block_task(
        self, output_volume, output_location, inputs):
        '''Create a new volume block from other volumes
        
        The input volumes are scanned and their overlaps are written to
        the output volume. If input volumes overlap and the overlap is in
        the output volume, the pixel values from the last input volume in
        the list are written to the output volume.

        :param output_volume: the volume coords of the output dataset
        :param output_location: where to store the output dataset
        :param inputs: a sequence of dictionaries having keys of
        "volume" and "location". Each "volume" is a
        :py:class:`ariadne_microns_pipeline.parameters.Volume` describing
        the volume of the input dataset. Each "location" is a
        :py:class:`ariadne_microns_pipeline.parameters.DatasetLocation`
        describing the location of the dataset on disk.
        '''
        #
        # Make the "inputs" variable hashable
        #
        inputs = to_hashable(inputs)
        return BlockTask(
            output_location=output_location,
            output_volume=output_volume,
            input_volumes=inputs)
    
    def gen_extract_dataset_task(self, in_hdf5_file, dataset_name):
        '''Extract an HDF5VolumeTarget or other dataset from an HDF5 file
        
        Given an HDF5FileTarget, extract one of its datasets.
        :param in_hdf5_file: the HDF5 file containing the dataset
        :param dataset_name: the name of the dataset within the HDF5 file
        '''
        return ExtractDatasetTask(in_hdf5_file.path, dataset_name)
    
    def gen_segmentation_statistics_task(self,
                                         volume,
                                         gt_seg_location,
                                         pred_seg_location,
                                         output_location):
        '''Collect statistics on the accuracy of a prediction
        
        This task does a statistical analysis of the accuracy of a prediction,
        comparing it against the ground truth. The data are saved as a
        JSON dictionary.
        
        :param volume: The volume that was segmented
        :param gt_seg_location: the location of the ground truth volume
        :param pred_seg_location: the location of the classifier prediction
        :param output_location: where to put the JSON file that contains the
        statistics.
        '''
        return SegmentationStatisticsTask(
            volume=volume,
            ground_truth_location = gt_seg_location,
            test_location=pred_seg_location,
            output_path=output_location)
    
    def gen_segmentation_report_task(self,
                                     csv_location,
                                     pdf_location):
        '''Generate a statistics report on the segmentation
        
        :param csv_location: the path to the CSV file containing the statistics
        :param pdf_location: where to write the matplotlib report
        '''
        return SegmentationReportTask(
            csv_location=csv_location,
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
    
