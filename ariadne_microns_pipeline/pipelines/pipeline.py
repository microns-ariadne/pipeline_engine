import luigi
import rh_logger
from .utilities import PipelineRunReportMixin
from ..tasks.factory import AMTaskFactory
from ..tasks.classify import ClassifyShimTask
from ..tasks.find_seeds import SeedsMethodEnum, Dimensionality
from ..targets.classifier_target import PixelClassifierTarget
from ..targets.hdf5_target import HDF5FileTarget
from ..targets.butterfly_target import ButterflyChannelTarget
from ..parameters import Volume, VolumeParameter, DatasetLocation
import numpy as np
import os
import tempfile
import sys

'''The name of the segmentation dataset within the HDF5 file'''
SEG_DATASET = "segmentation"

'''The name of the watershed seed datasets'''
SEEDS_DATASET = "seeds"

'''The name of the border mask datasets'''
MASK_DATASET = "mask"

'''The name of the image datasets'''
IMG_DATASET = "image"

'''The name of the membrane probability datasets'''
MEMBRANE_DATASET = "membrane"

'''The name of the neuroproofed segmentation datasets'''
NP_DATASET = "neuroproof"

'''The name of the ground-truth dataset for statistics computation'''
GT_DATASET = "gt"

'''The name of the predicted segmentation for statistics computation'''
PRED_DATASET = "pred"

'''The pattern for border datasets

parent - name of parent dataset, e.g. "membrane"
direction - the adjacency direction, e.g. "z"
'''
BORDER_DATASET_PATTERN = "{parent}_{direction}-border"


class PipelineTaskMixin:
    '''The Ariadne-Microns pipeline'''
    
    #output_path=luigi.Parameter(
    #    description="The path to the HDF5 file holding the results")
    experiment = luigi.Parameter(
        description="The Butterfly experiment that produced the dataset")
    sample = luigi.Parameter(
        description="The ID of the biological sample that was imaged")
    dataset = luigi.Parameter(
        description="The name of the volume that was imaged")
    channel = luigi.Parameter(
        description="The name of the channel from which we take data")
    gt_channel = luigi.Parameter(
        description="The name of the channel containing the ground truth")
    url = luigi.Parameter(
        description="The URL of the Butterfly REST endpoint")
    pixel_classifier_path = luigi.Parameter(
        description="Path to pickled pixel classifier")
    neuroproof_classifier_path = luigi.Parameter(
        description="Location of Neuroproof classifier")
    volume = VolumeParameter(
        description="The volume to segment")
    #########
    #
    # Optional parameters
    #
    #########
    block_width = luigi.IntParameter(
        description="Width of one of the processing blocks",
        default=2048)
    block_height = luigi.IntParameter(
        description="Height of one of the processing blocks",
        default=2048)
    block_depth = luigi.IntParameter(
        description="Number of planes in a processing block",
        default=2048)
    np_x_pad = luigi.IntParameter(
        description="The size of the border region for the Neuroproof merge "
        "of blocks to the left and right. The value is the amount of padding"
        " on each of the blocks.",
        default=30)
    np_y_pad = luigi.IntParameter(
        description="The size of the border region for the Neuroproof merge "
        "of blocks above and below. The value is the amount of padding"
        " on each of the blocks.",
        default=30)
    np_z_pad = luigi.IntParameter(
        description="The size of the border region for the Neuroproof merge "
        "of z-stacks. The value is the amount of padding"
        " on each of the blocks.",
        default=5)
    temp_dirs = luigi.ListParameter(
        description="The base location for intermediate files",
        default=(tempfile.gettempdir(),))
    membrane_class_name = luigi.Parameter(
        description="The name of the pixel classifier's membrane class",
        default="membrane")
    close_width = luigi.IntParameter(
        description="The width of the structuring element used for closing "
        "when computing the border masks.",
        default=5)
    sigma_xy = luigi.FloatParameter(
        description="The sigma in the X and Y direction of the Gaussian "
        "used for smoothing the probability map",
        default=3)
    sigma_z = luigi.FloatParameter(
        description="The sigma in the Z direction of the Gaussian "
        "used for smoothing the probability map",
        default=.4)
    threshold = luigi.FloatParameter(
        description="The threshold used during segmentation for finding seeds",
        default=1)
    method = luigi.EnumParameter(enum=SeedsMethodEnum,
        default=SeedsMethodEnum.Smoothing,
        description="The algorithm for finding seeds")
    dimensionality = luigi.EnumParameter(enum=Dimensionality,
        default=Dimensionality.D3,
        description="Whether to find seeds in planes or in a 3d volume")
    statistics_csv_path = luigi.Parameter(
        description="The path to the CSV statistics output file.",
        default="/dev/null")


    def get_dirs(self, x, y, z):
        '''Return a directory suited for storing a file with the given offset
        
        Create a hierarchy of directories in order to limit the number
        of files in any one directory.
        '''
        return [os.path.join(temp_dir,
                             self.experiment,
                             self.sample,
                             self.dataset,
                             self.channel,
                             str(x),
                             str(y),
                             str(z)) for temp_dir in self.temp_dirs]
    
    def get_pattern(self, dataset_name):
        return "{x:09d}_{y:09d}_{z:09d}_"+dataset_name
    
    def get_dataset_location(self, volume, dataset_name):
        return DatasetLocation(self.get_dirs(volume.x, volume.y, volume.z),
                               dataset_name,
                               self.get_pattern(dataset_name))
    
    def compute_extents(self):
        '''Compute various block boundaries and padding
        
        self.nn_{x,y,z}_pad - amount of padding for pixel classifier
        
        self.{x, y, z}{0, 1} - the start and end extents in the x, y & z dirs

        self.n_{x, y, z} - the number of blocks in the x, y and z dirs

        self.{x, y, z}s - the starts of each block (+1 at the end so that
        self.xs[n], self.xs[n+1] are the start and ends of block n)
        '''
        butterfly = ButterflyChannelTarget(
            self.experiment, self.sample, self.dataset, self.channel, 
            self.url)
        #
        # The useable width, height and depth are the true widths
        # minus the classifier padding
        #
        classifier = self.pixel_classifier.classifier
        self.nn_x_pad = classifier.get_x_pad()
        self.nn_y_pad = classifier.get_y_pad()
        self.nn_z_pad = classifier.get_z_pad()
        self.x1 = min(butterfly.x_extent - classifier.get_x_pad(), 
                      self.volume.x + self.volume.width)
        self.y1 = min(butterfly.y_extent - classifier.get_y_pad(),
                      self.volume.y + self.volume.height)
        self.z1 = min(butterfly.z_extent - classifier.get_z_pad(),
                      self.volume.z + self.volume.depth)
        self.x0 = max(classifier.get_x_pad(), self.volume.x)
        self.y0 = max(self.nn_y_pad, self.volume.y)
        self.z0 = max(self.nn_z_pad, self.volume.z)
        self.useable_width = self.x1 - self.x0
        self.useable_height = self.y1 - self.y0
        self.useable_depth = self.z1 - self.z0
        #
        # Compute equi-sized blocks (as much as possible)
        #
        self.n_x = int((self.useable_width-1) / self.block_width) + 1
        self.n_y = int((self.useable_height-1) / self.block_height) + 1
        self.n_z = int((self.useable_depth-1) / self.block_depth) + 1
        self.xs = np.linspace(self.x0, self.x1, self.n_x + 1).astype(int)
        self.ys = np.linspace(self.y0, self.y1, self.n_y + 1).astype(int)
        self.zs = np.linspace(self.z0, self.z1, self.n_z + 1).astype(int)

    def generate_butterfly_tasks(self):
        '''Get volumes padded for CNN'''
        self.butterfly_tasks = np.zeros((self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            z0, z1 = self.zs[zi] - self.nn_z_pad, self.zs[zi+1] + self.nn_z_pad
            for yi in range(self.n_y):
                y0 = self.ys[yi] - self.nn_y_pad
                y1 = self.ys[yi+1] + self.nn_y_pad
                for xi in range(self.n_x):
                    x0 = self.xs[xi] - self.nn_x_pad
                    x1 = self.xs[xi+1] + self.nn_x_pad
                    volume = Volume(x0, y0, z0, x1-x0, y1-y0, z1-z0)
                    location =self.get_dataset_location(volume, IMG_DATASET)
                    self.butterfly_tasks[zi, yi, xi] =\
                        self.factory.gen_get_volume_task(
                            experiment=self.experiment,
                            sample=self.sample,
                            dataset=self.dataset,
                            channel=self.channel,
                            url=self.url,
                            volume=volume,
                            location=location)

    def generate_classifier_tasks(self):
        '''Get the pixel classifier tasks
        
        Take each butterfly task and run a pixel classifier on its output.
        '''
        self.classifier_tasks = np.zeros((self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    btask = self.butterfly_tasks[zi, yi, xi]
                    input_target = btask.output()
                    img_location = DatasetLocation(
                        input_target.paths,
                        input_target.dataset_path,
                        input_target.pattern)
                    paths = self.get_dirs(self.xs[xi], self.ys[yi], self.zs[zi])
                    ctask = self.factory.gen_classify_task(
                        paths=paths,
                        datasets={self.membrane_class_name: MEMBRANE_DATASET},
                        pattern=self.get_pattern(MEMBRANE_DATASET),
                        img_volume=btask.volume,
                        img_location=img_location,
                        classifier_path=self.pixel_classifier_path)
                    ctask.set_requirement(btask)
                    #
                    # Create a shim that returns the membrane volume
                    # as its output.
                    #
                    shim_task = ClassifyShimTask.make_shim(
                        classify_task=ctask,
                        dataset_name=MEMBRANE_DATASET)
                    self.classifier_tasks[zi, yi, xi] = shim_task
    
    def generate_border_mask_tasks(self):
        '''Create a border mask for each block'''
        self.border_mask_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    ctask = self.classifier_tasks[zi, yi, xi]
                    input_target = ctask.output()
                    input_location = DatasetLocation(
                        input_target.paths,
                        input_target.dataset_path,
                        input_target.pattern)
                    volume = ctask.output_volume
                    location = self.get_dataset_location(volume, MASK_DATASET)
                    btask = self.factory.gen_mask_border_task(
                        volume,
                        input_location,
                        location,
                        border_width=self.np_x_pad,
                        close_width=self.close_width)
                    self.border_mask_tasks[zi, yi, xi] = btask
                    btask.set_requirement(ctask)

    def generate_seed_tasks(self):
        '''Find seeds for the watersheds'''
        self.seed_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    ctask = self.classifier_tasks[zi, yi, xi]
                    volume = ctask.volume
                    prob_target = ctask.output()
                    prob_location = DatasetLocation(
                        prob_target.paths,
                        prob_target.dataset_path,
                        prob_target.pattern)
                    seeds_location = \
                        self.get_dataset_location(volume, SEEDS_DATASET)
                    stask = self.factory.gen_find_seeds_task(
                        volume=volume,
                        prob_location=prob_location, 
                        seeds_location=seeds_location, 
                        sigma_xy=self.sigma_xy, 
                        sigma_z=self.sigma_z, 
                        threshold=self.threshold,
                        method=self.method,
                        dimensionality=self.dimensionality)
                    self.seed_tasks[zi, yi, xi] = stask
                    stask.set_requirement(ctask)

    def generate_watershed_tasks(self):
        '''Run watershed on each pixel '''
        self.watershed_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    ctask = self.classifier_tasks[zi, yi, xi]
                    btask = self.border_mask_tasks[zi, yi, xi]
                    seeds_task = self.seed_tasks[zi, yi, xi]
                    volume = btask.volume
                    prob_target = ctask.output()
                    prob_location = DatasetLocation(
                        prob_target.paths,
                        prob_target.dataset_path,
                        prob_target.pattern)
                    seeds_target = seeds_task.output()
                    seeds_location = seeds_target.dataset_location
                    seg_location = \
                        self.get_dataset_location(volume, SEG_DATASET)
                    stask = self.factory.gen_segmentation_task(
                        volume=btask.volume,
                        prob_location=prob_location,
                        mask_location=btask.mask_location,
                        seeds_location=seeds_location,
                        seg_location=seg_location,
                        sigma_xy=self.sigma_xy,
                        sigma_z=self.sigma_z)
                    self.watershed_tasks[zi, yi, xi] = stask
                    stask.set_requirement(ctask)
                    stask.set_requirement(btask)
                    stask.set_requirement(seeds_task)
    
    def generate_border_tasks(self):
        '''Create border cutouts between adjacent blocks
        
        Create both watershed and membrane probability blocks
        '''
        self.generate_z_border_tasks()
        self.generate_y_border_tasks()
        self.generate_x_border_tasks()
    
    def generate_border_task(self, a, b, out_volume, dataset_name):
        '''Generate a border task between two input tasks
        
        :param a: the first input task
        :param b: the second input task
        :param out_volume: the cutout volume
        :param dataset_name: the name of the dataset to be generated
        '''
        out_location = self.get_dataset_location(out_volume, dataset_name)
        inputs = []
        for ds in a.output(), b.output():
            volume = Volume(ds.x, ds.y, ds.z, ds.width, ds.height, ds.depth)
            location = DatasetLocation(ds.paths, ds.dataset_path, ds.pattern)
            inputs.append(dict(volume=volume, location=location))
        btask = self.factory.gen_block_task(
            output_location=out_location,
            output_volume=out_volume,
            inputs=inputs)
        btask.set_requirement(a)
        btask.set_requirement(b)
        return btask
    
    def generate_x_border_task(self, left, right, dataset_name):
        '''Generate a border task linking z-neighbors
        
        :param left: the task that generates the volume to the left
        :param right: the task that generates the volume to the right
        :param dataset_name: the name for the output dataset
        :returns: a block task whose output is a cutout of the border
        region between the two.
        '''
        ds_left = left.output()
        ds_right = right.output()
        x = ds_left.x + ds_left.width - self.np_x_pad
        width = self.np_x_pad * 2
        out_volume = Volume(
           x, ds_left.y, ds_left.z,
           width, ds_left.height, ds_left.depth)
        return self.generate_border_task(
            left, right, out_volume, dataset_name)
    
    def generate_y_border_task(self, top, bottom, dataset_name):
        '''Generate a border task linking y-neighbors
        
        :param top: the task that generates the upper volume
        :param bottoom: the task that generates the lower volume
        :param dataset_name: the name for the output dataset
        :returns: a block task whose output is a cutout of the border
        region between the two.
        '''
        ds_top = top.output()
        ds_bottom = bottom.output()
        y = ds_top.y + ds_top.height - self.np_y_pad
        height = self.np_y_pad * 2
        out_volume = Volume(
           ds_top.x, y, ds_top.z,
           ds_top.width, height, ds_top.depth)
        return self.generate_border_task(
            top, bottom, out_volume, dataset_name)

    def generate_z_border_task(self, above, below, dataset_name):
        '''Generate a border task linking z-neighbors
        
        :param above: the task that generates the upper volume
        :param below: the task that generates the lower volume
        :param dataset_name: the name for the output dataset
        :returns: a block task whose output is a cutout of the border
        region between the two.
        '''
        ds_above = above.output()
        ds_below = below.output()
        z = ds_above.z + ds_above.depth - self.np_z_pad
        depth = self.np_z_pad * 2
        out_volume = Volume(
           ds_above.x, ds_above.y, z,
           ds_above.width, ds_above.height, depth)
        return self.generate_border_task(
            above, below, out_volume, dataset_name)

    def generate_x_border_tasks(self):
        '''Create border cutouts between blocks left and right'''
        self.x_seg_borders = \
            np.zeros((self.n_z, self.n_y, self.n_x-1), object)
        self.x_prob_borders = \
            np.zeros((self.n_z, self.n_y, self.n_x-1), object)
        #
        # The task sets are composed of the input task arrays
        # the output border task arrays and the dataset name of
        # the input tasks
        #
        task_sets = ((self.classifier_tasks,
                      self.x_prob_borders,
                      MEMBRANE_DATASET),
                     (self.watershed_tasks,
                      self.x_seg_borders,
                      SEG_DATASET))
        
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xleft in range(self.n_x-1):
                    xright = xleft+1
                    for tasks_in, tasks_out, in_dataset_name in task_sets:
                    
                        task_left = tasks_in[zi, yi, xleft]
                        task_right = tasks_in[zi, yi, xright]
                        out_dataset_name = BORDER_DATASET_PATTERN.format(
                            parent=in_dataset_name, direction="x")
                        tasks_out[zi, yi, xleft] = \
                            self.generate_x_border_task(
                                task_left, task_right,
                                out_dataset_name)

    def generate_y_border_tasks(self):
        '''Create border cutouts between blocks on top and below'''
        self.y_seg_borders = \
            np.zeros((self.n_z, self.n_y-1, self.n_x), object)
        self.y_prob_borders = \
            np.zeros((self.n_z, self.n_y-1, self.n_x), object)
        #
        # The task sets are composed of the input task arrays
        # the output border task arrays and the dataset name of
        # the input tasks
        #
        task_sets = ((self.classifier_tasks,
                      self.y_prob_borders,
                      MEMBRANE_DATASET),
                     (self.watershed_tasks,
                      self.y_seg_borders,
                      SEG_DATASET))
        
        for zi in range(self.n_z):
            for ytop in range(self.n_y-1):
                ybottom = ytop+1
                for xi in range(self.n_x):
                    for tasks_in, tasks_out, in_dataset_name in task_sets:
                    
                        task_top = tasks_in[zi, ytop, xi]
                        task_bottom = tasks_in[zi, ybottom, xi]
                        out_dataset_name = BORDER_DATASET_PATTERN.format(
                            parent=in_dataset_name, direction="y")
                        tasks_out[zi, ytop, xi] = \
                            self.generate_y_border_task(
                                task_top, task_bottom,
                                out_dataset_name)

    def generate_z_border_tasks(self):
        '''Create border cutouts between blocks above and below'''
        self.z_seg_borders = \
            np.zeros((self.n_z-1, self.n_y, self.n_x), object)
        self.z_prob_borders = \
            np.zeros((self.n_z-1, self.n_y, self.n_x), object)
        #
        # The task sets are composed of the input task arrays
        # the output border task arrays and the dataset name of
        # the input tasks
        #
        task_sets = ((self.classifier_tasks,
                      self.z_prob_borders,
                      MEMBRANE_DATASET),
                     (self.watershed_tasks,
                      self.z_seg_borders,
                      SEG_DATASET))
        
        for zabove in range(self.n_z-1):
            zbelow = zabove + 1
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    for tasks_in, tasks_out, in_dataset_name in task_sets:
                    
                        task_above = tasks_in[zabove, yi, xi]
                        task_below = tasks_in[zbelow, yi, xi]
                        out_dataset_name = BORDER_DATASET_PATTERN.format(
                            parent=in_dataset_name, direction="z")
                        tasks_out[zabove, yi, xi] = \
                            self.generate_z_border_task(
                                task_above, task_below,
                                out_dataset_name)
                        
    def generate_neuroproof_tasks(self):
        '''Generate all tasks involved in Neuroproofing a segmentation
        
        We Neuroproof the blocks and the x, y and z borders between blocks
        '''
        self.np_tasks = np.zeros((self.n_z, self.n_y, self.n_x), object)
        self.np_x_border_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x-1), object)
        self.np_y_border_tasks = \
            np.zeros((self.n_z, self.n_y-1, self.n_x), object)
        self.np_z_border_tasks = \
            np.zeros((self.n_z-1, self.n_y, self.n_x), object)
        #
        # The task sets are composed of
        # classifier tasks
        # segmentation tasks
        # output tasks
        # output dataset name
        # is_border: True if neuroproofing a border, False if neuroproofing
        #            a block
        # 
        task_sets = (
            (self.classifier_tasks,
             self.watershed_tasks,
             self.np_tasks,
             NP_DATASET,
             False),
            (self.x_prob_borders,
             self.x_seg_borders,
             self.np_x_border_tasks,
             BORDER_DATASET_PATTERN.format(parent=NP_DATASET, direction="x"),
             True),
            (self.y_prob_borders,
             self.y_seg_borders,
             self.np_y_border_tasks,
             BORDER_DATASET_PATTERN.format(parent=NP_DATASET, direction="y"),
             True),
            (self.z_prob_borders,
             self.z_seg_borders,
             self.np_z_border_tasks,
             BORDER_DATASET_PATTERN.format(parent=NP_DATASET, direction="z"),
             True)
        )
        for classifier_tasks, seg_tasks, np_tasks, dataset_name, is_border\
            in task_sets:

            for zi in range(classifier_tasks.shape[0]):
                for yi in range(classifier_tasks.shape[1]):
                    for xi in range(classifier_tasks.shape[2]):
                        classifier_task = classifier_tasks[zi, yi, xi]
                        seg_task = seg_tasks[zi, yi, xi]
                        volume = classifier_task.output_volume
                        output_seg_location = self.get_dataset_location(
                            volume, dataset_name)
                        np_task = self.factory.gen_neuroproof_task(
                            volume=volume,
                            prob_location=classifier_task.output_location,
                            input_seg_location=seg_task.output_location,
                            output_seg_location=output_seg_location,
                            classifier_filename=self.neuroproof_classifier_path)
                        np_task.set_requirement(classifier_task)
                        np_task.set_requirement(seg_task)
                        np_tasks[zi, yi, xi] = np_task
    
    def generate_gt_cutouts(self):
        '''Generate volumes of ground truth segmentation'''
        
        self.gt_block_tasks = np.zeros((self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            z0 = self.zs[zi] + self.np_z_pad
            z1 = self.zs[zi+1] - self.np_z_pad
            for yi in range(self.n_y):
                y0 = self.ys[yi] + self.np_y_pad
                y1 = self.ys[yi+1] - self.np_y_pad
                for xi in range(self.n_x):
                    x0 = self.xs[xi] + self.np_x_pad
                    x1 = self.xs[xi+1] - self.np_x_pad
                    volume=Volume(x0, y0, z0, x1-x0, y1-y0, z1-z0)
                    dataset_location = self.get_dataset_location(
                        volume, GT_DATASET)
                    btask = self.factory.gen_get_volume_task(
                        self.experiment,
                        self.sample,
                        self.dataset,
                        self.gt_channel,
                        self.url,
                        volume,
                        dataset_location)
                    self.gt_block_tasks[zi, yi, xi] = btask
    
    def generate_pred_cutouts(self):
        '''Generate volumes matching the ground truth segmentations'''
        self.pred_block_tasks = np.zeros((self.n_z, self.n_y, self.n_x),
                                         object)
        for zi in range(self.n_z):
            z0 = self.zs[zi] + self.np_z_pad
            z1 = self.zs[zi+1] - self.np_z_pad
            for yi in range(self.n_y):
                y0 = self.ys[yi] + self.np_y_pad
                y1 = self.ys[yi+1] - self.np_y_pad
                for xi in range(self.n_x):
                    x0 = self.xs[xi] + self.np_x_pad
                    x1 = self.xs[xi+1] - self.np_x_pad
                    volume=Volume(x0, y0, z0, x1-x0, y1-y0, z1-z0)
                    dataset_location = self.get_dataset_location(
                        volume, PRED_DATASET)
                    nptask = self.np_tasks[zi, yi, xi]
                    btask = self.factory.gen_block_task(
                        volume, dataset_location,
                        [dict(volume=nptask.volume,
                              location=nptask.output_seg_location)])
                    btask.set_requirement(nptask)
                    self.pred_block_tasks[zi, yi, xi] = btask
                    
    def generate_statistics_tasks(self):
        if self.statistics_csv_path != "/dev/null":
            self.statistics_tasks = np.zeros((self.n_z, self.n_y, self.n_x),
                                             object)
            self.generate_gt_cutouts()
            self.generate_pred_cutouts()
            json_paths = []
            for zi in range(self.n_z):
                for yi in range(self.n_y):
                    for xi in range(self.n_x):
                        ptask = self.pred_block_tasks[zi, yi, xi]
                        gttask = self.gt_block_tasks[zi, yi, xi]
                        output_location = os.path.join(
                            self.get_dirs(
                                self.xs[xi], self.ys[yi], self.zs[zi])[0],
                            "segmentation_statistics.json")
                        stask = self.factory.gen_segmentation_statistics_task(
                            volume=ptask.output_volume, 
                            gt_seg_location=gttask.destination,
                            pred_seg_location=ptask.output_location,
                            output_location=output_location)
                        stask.set_requirement(ptask)
                        stask.set_requirement(gttask)
                        self.statistics_tasks[zi, yi, xi] = stask
                        output_target = stask.output()
                        output_target.is_tmp=True
                        json_paths.append(output_target.path)
            self.statistics_csv_task = self.factory.gen_json_to_csv_task(
                json_paths=json_paths,
                output_path = self.statistics_csv_path)
            for stask in self.statistics_tasks.flatten():
                self.statistics_csv_task.set_requirement(stask)
            pdf_path = os.path.splitext(self.statistics_csv_path)[0] + ".pdf"
            self.statistics_report_task = \
                self.factory.gen_segmentation_report_task(
                    self.statistics_csv_task.output().path,
                    pdf_path)
            self.statistics_report_task.set_requirement(
                self.statistics_csv_task)
        else:
            self.statistics_csv_task = None
    
    def compute_requirements(self):
        '''Compute the requirements for this task'''
        if not hasattr(self, "requirements"):
            try:
                rh_logger.logger.report_event("Assembling pipeline")
            except:
                rh_logger.logger.start_process("Ariadne pipeline",
                                               "Assembling pipeline")
            try:
                self.factory = AMTaskFactory()
                rh_logger.logger.report_event(
                    "Loading pixel classifier")
                self.pixel_classifier = PixelClassifierTarget(
                    self.pixel_classifier_path)
                rh_logger.logger.report_event(
                    "Computing blocks")
                self.compute_extents()
                #
                # Step 1: get data from Butterfly
                #
                rh_logger.logger.report_event("Making Butterfly download tasks")
                self.generate_butterfly_tasks()
                #
                # Step 2: run the pixel classifier on each
                #
                rh_logger.logger.report_event("Making classifier tasks")
                self.generate_classifier_tasks()
                #
                # Step 3: make the border masks
                #
                rh_logger.logger.report_event("Making border mask tasks")
                self.generate_border_mask_tasks()
                #
                # Step 4: find the seeds for the watershed
                #
                rh_logger.logger.report_event("Making watershed seed tasks")
                self.generate_seed_tasks()
                #
                # Step 5: run watershed
                #
                rh_logger.logger.report_event("Making watershed tasks")
                self.generate_watershed_tasks()
                #
                # Step 6: create all the border blocks
                #
                rh_logger.logger.report_event("Making border reblocking tasks")
                self.generate_border_tasks()
                #
                # Step 7: run Neuroproof on the blocks and border blocks
                #
                rh_logger.logger.report_event("Making Neuroproof tasks")
                self.generate_neuroproof_tasks()
                #
                # TO DO: skeletonization and the rest of rollup. For now
                #        our output is the neuroproofed segmentation
                #
                self.requirements =\
                    self.np_tasks.flatten().tolist() +\
                    self.np_x_border_tasks.flatten().tolist() +\
                    self.np_y_border_tasks.flatten().tolist() +\
                    self.np_z_border_tasks.flatten().tolist()
                #
                # (maybe) generate the statistics tasks
                #
                self.generate_statistics_tasks()
                if self.statistics_csv_task is not None:
                    self.requirements.append(self.statistics_report_task)
                rh_logger.logger.report_event(
                    "Pipeline task graph computation finished")
            except:
                rh_logger.logger.report_exception()
                raise
    
    def requires(self):
        self.compute_requirements()
        return self.requirements
        
    #def output(self):
    #    return HDF5FileTarget(self.output_path,
    #                          [SEG_DATASET])


class PipelineTask(PipelineTaskMixin, PipelineRunReportMixin, luigi.Task):
    task_namespace = "ariadne_microns_pipeline"
    
        
        
        