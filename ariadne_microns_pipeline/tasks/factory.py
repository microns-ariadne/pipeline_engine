'''The task factory creates tasks for particular pipeline steps.'''

import json
import luigi
from download_from_butterfly import DownloadFromButterflyTask
from block import BlockTask
from extract_dataset import ExtractDatasetTask
from classify import PixelClassifierTask

class AMTaskFactory(object):
    '''Factory for creating Ariadne/Microns tasks
    
    Each method has its output target first and input targets following,
    e.g.
    
    def gen_foo_from_bar_and_baz(foo, bar, baz)
    
    The return value is the task that performs the action
    '''
   
    def gen_get_volume_task(self,
                            paths,
                            dataset_path,
                            pattern,
                            experiment,
                            sample,
                            dataset,
                            channel,
                            url,
                            x, y, z, width, height, depth):
        '''Get a 3d volume
        
        :param paths: the paths for sharding to different spindles
        :param dataset_path: the name of the target volume dataset
        :param pattern: the pattern for filenames
        :param experiment: the experiment done to produce the sample
        :param sample: the sample ID of the tissue that was imaged
        :param dataset: the volume that was imaged
        :param channel: the channel supplying the pixel values
        :param url: the URL of the butterfly server
        :param x: the x-offset of the volume within the dataset
        :param y: the y-offset of the volume within the dataset
        :param z: the z-offset of the volume within the dataset
        :param width: the width of the volume
        :param height: the height of the volume
        :param depth: the depth of the volume
        '''
        
        return DownloadFromButterflyTask(dest_paths=path,
                                         dest_dataset_path=dataset_path,
                                         dest_pattern=pattern,
                                         experiment=experiment,
                                         sample=sample,
                                         dataset=dataset,
                                         channel=channel,
                                         url=url,
                                         x=x,
                                         y=y,
                                         z=z,
                                         width=width,
                                         height=height,
                                         depth=depth)
    
    def gen_classify_task(self, path, datasets, img_volume, classifier):
        '''Classify a volume

        :param path: the path to the HDF5 file that will hold the
             probability volumes
        :param datasets: a dictionary with keys of the class indexes or names
             produced by the classifier and values of the names of the
             datasets to be stored (not all datasets from the classifier need
             be stored)
        :param img_volume: the image to be classified
        :param classifier: a trained classifier
        '''
        raise NotImplementedError()
    
    def gen_watershed_task(self, path, dataset, landscape_volume, seeds):
        '''Run a watershed

        Run a seeded watershed, producing a segmentation from a landscape
        of hills and valleys and a list of 3-d seed values
        
        :param path: the path to the HDF5 file holding the segmentation produced
        :param dataset: the name of the dataset holding the segmentation
        :param landscape_volume: the landscape to be watershedded
        :param seeds: the seeds marking the places to start the watershed
        '''
        raise NotImplementedError()

    def gen_neuroproof_task(self, path, dataset, seg_volume, landscape_volume, 
                            classifier):
        '''Run Neuroproof on an oversegmented volume
        
        :param path: the path to the HDF5 file containing the merge dataset
        :param dataset: the dataset name of the output of Neuroproof, 
            a volume of potentially merged segments from the input segmentation
        :param seg_volume: the oversegmented input volume
        :param landscape_volume: the watershed landscape that was an
            input for the oversegmentation
        :param classifier: the classifier trained to assess merge/no merge
        decisions.
        '''
        raise NotImplementedError()
    
    def gen_mask_border_task(self,
                             path,
                             outer_mask_name,
                             border_mask_name,
                             all_outer_mask_name,
                             all_border_mask_name,
                             img_volume, border_width, close_width):
        '''Generate the outer and border masks for a volume
        
        :param path: The path to the HDF5 file containing the masks
        :param outer_mask_name: the name of the outer mask dataset
        :param border_mask_dataset: the name of the border mask dataset
        :param all_outer_mask_name: the name of the outer mask projection
        :param all_border_mask_name: the name of the border mask projection
        :param img_volume: The volume of the image which will have its mask
            calculated.
        :param border_width: The width of the border to consider masking
        :param close_width: the size of the square block to be used in the
             morphological closing operation.
        '''
        raise NotImplementedError()
    
    def gen_apply_mask_task(self, path, dataset, masks, in_volume):
        '''Apply the masks to the input volume
        
        :param path: the path to the masked volume's HDF5 file
        :param dataset: the name of the dataset within the HDF5 file
        :param masks: the masks to be applied
        :param in_volume: the volume to be masked
        '''
        raise NotImplementedError()
    
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
    
    def gen_pixel_classifier_task(self, classifier_path):
        '''Unpickle a pixel classifier
        
        The output of this task is a PixelClassifierTarget whose classifier
        can be used to get class probabilities for pixels in an image.
        
        :param classifier_path: the path to the pickle of the classifier
        '''
        return PixelClassifierTask(classifier_path)
    
    #########################
    #
    # Helper tasks
    #
    #    These help bridge one task to another.
    #
    #########################

    def gen_block_task(self,
                       path, 
                       dataset_path, 
                       x, y, z,
                       width, height, depth, 
                       in_volumes):
        '''Create a new volume block from other volumes
        
        The input volumes are scanned and their overlaps are written to
        the output volume. If input volumes overlap and the overlap is in
        the output volume, the pixel values from the last input volume in
        the list are written to the output volume.
        
        :param path: the path of the output volume's HDF5 file
        :param dataset_path: the name of the dataset within the HDF5 file
        :param x: the x offset of the output volume
        :param y: the y offset of the output volume
        :param z: the z offset of the output volume
        :param width: the width of the output volume
        :param height: the height of the output volume
        :param in_volumes: a sequence of input volumes
        '''
        in_volume_list = [(_.path, _.dataset_path) for _ in in_volumes]
        return BlockTask(
            path,
            dataset_path,
            input_volumes=json.dumps(in_volume_list),
            x=x,
            y=y,
            z=z,
            width=width,
            height=height,
            depth=depth)
    
    def gen_extract_dataset_task(self, in_hdf5_file, dataset_name):
        '''Extract an HDF5VolumeTarget or other dataset from an HDF5 file
        
        Given an HDF5FileTarget, extract one of its datasets.
        :param in_hdf5_file: the HDF5 file containing the dataset
        :param dataset_name: the name of the dataset within the HDF5 file
        '''
        return ExtractDatasetTask(in_hdf5_file.path, dataset_name)