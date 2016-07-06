import luigi
from factory import AMTaskFactory
from ..targets.hdf5_target import HDF5FileTarget
from ..targets.butterfly_target import ButterflyChannelTarget
from classify import PixelClassifierTask
import numpy as np
import os
import tempfile
import sys

'''The name of the segmentation dataset within the HDF5 file'''
SEG_DATASET = "segmentation"
IMG_DATASET = "image"

class PipelineTaskMixin:
    '''The Ariadne-Microns pipeline'''
    
    output_path=luigi.Parameter(
        description="The path to the HDF5 file holding the results")
    experiment = luigi.Parameter(
        description="The Butterfly experiment that produced the dataset")
    sample = luigi.Parameter(
        description="The ID of the biological sample that was imaged")
    dataset = luigi.Parameter(
        description="The name of the volume that was imaged")
    channel = luigi.Parameter(
        description="The name of the channel from which we take data")
    url = luigi.Parameter(
        description="The URL of the Butterfly REST endpoint")
    pixel_classifier_path = luigi.Parameter(
        description="Path to pickled pixel classifier")
    neuroproof_classifier_path = luigi.Parameter(
        description="Location of Neuroproof classifier")
    #########
    #
    # Optional parameters
    #
    #########
    x = luigi.IntParameter(
        description="The left side of the volume to segment",
        default=0)
    y = luigi.IntParameter(
        description="The top coordinate of the volume to segment",
        default=0)
    z = luigi.IntParameter(
        description="The first plane of the volume to segment",
        default=0)
    width = luigi.IntParameter(
        description="The width of the volume to segment",
        default=sys.maxint)
    height = luigi.IntParameter(
        description="The height of the volume to segment",
        default=sys.maxint)
    depth = luigi.IntParameter(
        description="The depth of the volume to segment",
        default=sys.maxint)
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
    np_y_pad = luigi.IntParameter(
        description="The size of the border region for the Neuroproof merge "
        "of z-stacks. The value is the amount of padding"
        " on each of the blocks.",
        default=5)
    temp_dirs = luigi.ListParameter(
        description="The base location for intermediate files",
        default=tempfile.gettempdir())
    membrane_class_name = luigi.Parameter(
        description="The name of the pixel classifier's membrane class",
        default="membrane"),
    close_width = luigi.IntParameter(
        description="The width of the structuring element used for closing "
        "when computing the border masks.",
        default=5)

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
        classifier = self.pixel_classifier.output()
        self.nn_x_pad = classifier.x_pad
        self.nn_y_pad = classifier.y_pad
        self.nn_z_pad = classifier.z_pad
        self.x1 = min(butterfly.x_extent - classifier.x_pad, 
                      self.x + self.width)
        self.y1 = min(butterfly.y_extent - classifier.y_pad,
                      self.y + self.height)
        self.z1 = min(butterfly.z_extent - classifier.z_pad,
                      self.z + self.depth)
        self.x0 = max(classifier.x_pad, self.x)
        self.y0 = max(self.nn_y_pad, self.y)
        self.z0 = max(self.nn_z_pad, self.z)
        self.useable_width = x1 - x0
        self.useable_height = y1 - y0
        self.useable_depth = z1 - z0
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
                    self.butterfly_tasks[zi, yi, xi] =\
                        self.factory.gen_get_volume_task(
                            paths=self.get_dirs(x0, y0, z0),
                            dataset_path=IMG_DATASET,
                            experiment=self.experiment,
                            sample=self.sample,
                            dataset=self.dataset,
                            channel=self.channel,
                            url=self.url,
                            x=x0, y=y0, z=z0,
                            width=x1-x0,
                            height=y1-y0,
                            depth=z1-z0)
                    yield self.butterfly_tasks[zi, yi, xi]

    def generate_classifier_tasks(self):
        '''Get the pixel classifier tasks
        
        Take each butterfly task and run a pixel classifier on its output.
        '''
        self.classifier_tasks = np.zeros((self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    btask = self.butterfly_tasks[zi, yi, xi]
                    ctask = self.factory.gen_classify_task(
                        paths=self.temp_dirs,
                        datasets={self.membrane_class_name: "membrane"},
                        patterns={
                            self.membrane_class_name: 
                            "{x:09d}_{y:09d}_{z:09d}"},
                        img_volume=btask.output(), 
                        classifier=self.pixel_classifier.output())
                    self.classifier_tasks[zi, yi, xi] = ctask
                    yield ctask
    
    def generate_border_mask_tasks(self):
        '''Create a border mask for each block'''
        self.border_mask_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    ctask = self.classifier_tasks[zi, yi, xi]
                    btask = self.factory.gen_mask_border_task(
                        paths=self.temp_dirs,
                        pattern="{x:09d}_{y:09d}_{z:09d}",
                        outer_mask_name="outer",
                        border_mask_name="border",
                        all_outer_mask_name="all_outer",
                        all_border_mask_name="all_border",
                        img_volume=ctask.output(),
                        border_width=self.np_x_pad,
                        close_width=self.close_width)
                    self.border_mask_tasks[zi, yi, xi] = btask
                    yield btask
                    
    def generate_watershed_tasks(self):
        '''Run watershed on each pixel '''
        raise NotImplementedError()
    
    def requires(self):
        '''Return the tasks we need to complete this'''
        self.factory = AMTaskFactory()
        self.pixel_classifier = PixelClassifierTask()
        yield self.pixel_classifier
        self.compute_extents()
        #
        # Step 1: get data from Butterfly
        #
        for task in self.generate_butterfly_tasks():
            yield task
        #
        # Step 2: run the pixel classifier on each
        #
        for task in self.generate_classifier_tasks():
            yield task
        #
        # Step 3: make the border masks
        #
        for task in self.generate_border_mask_tasks():
            yield task
        #
        # Step 4: run watershed
        #
        for task in self.generate_watershed_tasks():
            yield task
        
        
        
    def output(self):
        return HDF5FileTarget(self.output_path,
                              [SEG_DATASET])


class PipelineRunMixin:
    
    def ariadne_run(self):
        
        
        