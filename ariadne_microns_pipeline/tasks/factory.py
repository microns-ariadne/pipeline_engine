'''The task factory creates tasks for particular pipeline steps.'''

import json
import luigi
import os
import rh_config

from .download_from_butterfly import DownloadFromButterflyTask
from .block import BlockTask
from .classify import ClassifyTask
from .json_to_csv_task import JSONToCSVTask
from .mask import MaskBorderTask
from .neuroproof import NeuroproofTask
from .segment import SegmentTask
from .segmentation_statistics import \
     SegmentationStatisticsTask, SegmentationReportTask
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
    
    def gen_segmentation_task(
        self, volume, prob_location, mask_location, seg_location,
        sigma_xy=None, sigma_z=None, threshold=None):
        '''Generate a segmentation task

        Generate a segmentation task.  The task takes a probability map of
        the membranes and a mask of areas to exclude from segmentation. It
        smooths the membrane probabilities with an anisotropic Gaussian
        (different sigmas in XY and Z), thresholds to get
        the seeds for the watershed, then performs a 3d watershed.

        :param volume: the volume to be segmented in global coordinates
        :param prob_location: where to find the membrane probability volume
        :param mask_location: where to find the mask location
        :param seg_location: where to put the segmentation
        :param sigma_xy: the sigma of the smoothing gaussian in the X and Y
        directions
        :param sigma_z: the sigma of the smoothing gaussian in the Z direction
        :param threshold: the threshold cutoff for finding seeds
        '''
        kwargs = {}
        if sigma_xy is not None:
            kwargs["sigma_xy"] = sigma_xy
        if sigma_z is not None:
            kwargs["sigma_z"] = sigma_z
        if threshold is not None:
            kwargs["threshold"] = threshold
        return SegmentTask(volume=volume, 
                           prob_location=prob_location,
                           mask_location=mask_location,
                           output_location=seg_location,
                           **kwargs)

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
        try:
            config = rh_config.config["neuroproof"]
            neuroproof = config["neuroproof_graph_predict"]
        except KeyError:
            raise ValueError(
                "The .rh_config.yaml file is missing configuration information "
                "for Neuroproof. See README.md for details on how to build and "
                "configure Neuroproof.")
        ld_library_path = os.pathsep.join(config.get("ld_library_path", []))
        return NeuroproofTask(volume=volume,
                              prob_location=prob_location,
                              input_seg_location=input_seg_location,
                              output_seg_location=output_seg_location,
                              neuroproof=neuroproof,
                              neuroproof_ld_library_path=ld_library_path,
                              classifier_filename=classifier_filename)
    
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
    
    def gen_run_pipeline_task(self,
                              seg_volume_path,
                              experiment,
                              sample,
                              dataset,
                              channel):
        '''Produce a segmentation of the volume

        :param seg_volume_path: The HDF5 file holding the volume
        :param experiment: The experiment that resulted in the sample
        :param sample: the sample ID of the tissue that was imaged
        :param dataset: the volume that was imaged
        :param channel: the channel on which the pixel values were taken
        '''
        raise NotImplementedError()
    
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
                             output_path):
        '''Collect a number of JSON dictionaries into one .csv file
        
        Each JSON dictionary is a row in the CSV files and the columns are
        the keys in the dictionary, sorted alphabetically.
        
        :param json_paths: a sequence of pathnames to the JSON input files
        :param output_path: the location for the .csv file
        '''
        return JSONToCSVTask(json_paths=json_paths,
                             output_path=output_path)