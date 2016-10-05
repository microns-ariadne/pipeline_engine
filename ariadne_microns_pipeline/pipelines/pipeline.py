import luigi
import rh_logger
from .utilities import PipelineRunReportMixin
from ..tasks.factory import AMTaskFactory
from ..tasks.classify import ClassifyShimTask
from ..tasks.find_seeds import SeedsMethodEnum, Dimensionality
from ..tasks.match_synapses import MatchMethod
from ..targets.classifier_target import PixelClassifierTarget
from ..targets.hdf5_target import HDF5FileTarget
from ..targets.butterfly_target import ButterflyChannelTarget
from ..parameters import Volume, VolumeParameter, DatasetLocation
from ..parameters import EMPTY_DATASET_LOCATION
from ..pipelines.synapse_gt_pipeline import SynapseGtTask
import numpy as np
import os
import tempfile
import sys

'''The name of the segmentation dataset within the HDF5 file'''
SEG_DATASET = "segmentation"

'''The name of the synapse segmentation dataset'''
SYN_SEG_DATASET = "synapse-segmentation"

'''The name of the filtered (by area) synapse segmentation dataset'''
FILTERED_SYN_SEG_DATASET = "filtered-synapse-segmentation"

'''The name of the 2D resegmentation of the 3d segmentation'''
RESEG_DATASET = "resegmentation"

'''The name of the watershed seed datasets'''
SEEDS_DATASET = "seeds"

'''The name of the border mask datasets'''
MASK_DATASET = "mask"

'''The name of the image datasets'''
IMG_DATASET = "image"

'''The name of the membrane probability datasets'''
MEMBRANE_DATASET = "membrane"

'''The name of the synapse probability datasets'''
SYNAPSE_DATASET = "synapse"

'''The name of the neuroproofed segmentation datasets'''
NP_DATASET = "neuroproof"

'''The name of the ground-truth dataset for statistics computation'''
GT_DATASET = "gt"

'''The name of the sub-block ground-truth dataset'''
GT_BLOCK_DATASET = "gt-block"

'''The name of the synapse gt dataset'''
SYN_GT_DATASET = "synapse-gt"

'''The name of the ground-truth annotation mask dataset'''
GT_MASK_DATASET = "gt-mask"

'''The name of the segmentation of the synapse gt dataset'''
SYN_SEG_GT_DATASET = "synapse-gt-segmentation"

'''The name of the predicted segmentation for statistics computation'''
PRED_DATASET = "pred"

'''The name of the skeleton directory'''
SKEL_DIR_NAME = "skeleton"

'''The pattern for border datasets

parent - name of parent dataset, e.g. "membrane"
direction - the adjacency direction, e.g. "z"
'''
BORDER_DATASET_PATTERN = "{parent}_{direction}-border"

'''The pattern for connected_components tasks

direction - the adjacency direction: x-, x+, y-, y+, z-, z+
'''
CONNECTED_COMPONENTS_PATTERN = "connected-components_{direction}.json"

'''Signals that the channel isn't available (e.g. no ground truth)'''
NO_CHANNEL = "no-channel"

class PipelineTaskMixin:
    '''The Ariadne-Microns pipeline'''
    
    experiment = luigi.Parameter(
        description="The Butterfly experiment that produced the dataset")
    sample = luigi.Parameter(
        description="The ID of the biological sample that was imaged")
    dataset = luigi.Parameter(
        description="The name of the volume that was imaged")
    channel = luigi.Parameter(
        default="raw",
        description="The name of the channel from which we take data")
    gt_channel = luigi.Parameter(
        default="gt",
        description="The name of the channel containing the ground truth")
    gt_mask_channel = luigi.Parameter(
        default=NO_CHANNEL,
        description="The name of the channel containing the mask indicating "
        "the volume that is annotated with ground-truth")
    synapse_channel = luigi.Parameter(
        default="synapse",
        description="The name of the channel containing ground truth "
        "synapse data")
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
    synapse_class_name = luigi.Parameter(
        description="The name of the pixel classifier's synapse class",
        default="synapse")
    additional_neuroproof_channels = luigi.ListParameter(
        default=[],
        description="The names of additional classifier classes "
                    "that are fed into Neuroproof as channels")
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
        description="The threshold used during segmentation for finding seeds"
        " or for thresholding membrane in connected components",
        default=1)
    method = luigi.EnumParameter(enum=SeedsMethodEnum,
        default=SeedsMethodEnum.Smoothing,
        description="The algorithm for finding seeds")
    dimensionality = luigi.EnumParameter(enum=Dimensionality,
        default=Dimensionality.D3,
        description="Whether to find seeds in planes or in a 3d volume")
    minimum_distance_xy = luigi.FloatParameter(
        default=5,
        description="The minimum distance allowed between seeds")
    minimum_distance_z = luigi.FloatParameter(
        default=1.5,
        description="The minimum distance allowed between seed in the z dir")
    statistics_csv_path = luigi.Parameter(
        description="The path to the CSV statistics output file.",
        default="/dev/null")
    synapse_statistics_path = luigi.Parameter(
        default="/dev/null",
        description=
        "The path to the .json synapse connectivity statistics file")
    wants_skeletonization = luigi.BoolParameter(
        description="Skeletonize the Neuroproof segmentation",
        default=False)
    wants_resegmentation = luigi.BoolParameter(
        description="Convert the 3D segmentation to 2D before Neuroproof",
        default=False)
    use_min_contact = luigi.BoolParameter(
        default=False,
        description="Break an object between two planes with a minimum of "
        "contact")
    contact_threshold = luigi.IntParameter(
        default=100,
        description="Break objects with less than this number of area overlap")
    connectivity_graph_location = luigi.Parameter(
        default="/dev/null",
        description="The location of the all-connected-components connectivity"
                    " .json file. Default = do not generate it.")
    min_percent_connected = luigi.FloatParameter(
         default=75.0,
         description="Minimum overlap required to join segments across blocks")
    #
    # NB: minimum synapse area in AC3 was 561, median was ~5000
    #
    min_synapse_area = luigi.IntParameter(
        description="Minimum area for a synapse",
        default=250)
    synapse_xy_erosion = luigi.IntParameter(
        default=4,
        description = "# of pixels to erode the neuron segmentation in the "
                      "X and Y direction prior to synapse segmentation.")
    synapse_z_erosion = luigi.IntParameter(
        default=1,
        description = "# of pixels to erode the neuron segmentation in the "
                      "Z direction prior to synapse segmentation.")
    synapse_xy_sigma = luigi.FloatParameter(
        description="Sigma for smoothing Gaussian for synapse segmentation "
                     "in the x and y directions.",
        default=3)
    synapse_z_sigma = luigi.FloatParameter(
        description="Sigma for smoothing Gaussian for symapse segmentation "
                     "in the z direction.",
        default=.5)
    synapse_min_size_2d = luigi.IntParameter(
        default=25,
        description="Remove isolated synapse foreground in a plane if "
        "less than this # of pixels")
    synapse_max_size_2d = luigi.IntParameter(
        default=10000,
        description = "Remove large patches of mislabeled synapse in a plane "
        "that have an area greater than this")
    synapse_min_size_3d = luigi.IntParameter(
        default=500,
        description = "Minimum size in voxels of a synapse")
    min_synapse_depth = luigi.IntParameter(
        default=3,
        description="Minimum acceptable size of a synapse in the Z direction")
    synapse_threshold = luigi.FloatParameter(
        description="Threshold for synapse voxels vs background voxels",
        default=128.)
    #
    # parameters for synapse connection
    #
    synapse_xy_dilation = luigi.IntParameter(
        description="How much to dilate the synapse segmentation in the "
                    "x and y direction before matching with neurons.",
        default=3)
    synapse_z_dilation = luigi.IntParameter(
        description="How much to dilate the synapse segmentation in the z "
                    "direction before matching with neurons.",
        default=0)
    min_synapse_neuron_contact = luigi.IntParameter(
        description="The minimum number of overlapping voxels needed "
                    "to consider joining neuron to synapse",
        default=25)
    #
    # parameters for synapse statistics
    #
    synapse_match_method = luigi.EnumParameter(
        enum=MatchMethod,
        default=MatchMethod.overlap,
        description="Method for matching detected synapses against "
        "ground-truth synapses")

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
    
    @property
    def wants_connectivity(self):
        '''True if we are doing a connectivity graph'''
        return self.connectivity_graph_location != "/dev/null"
    
    @property
    def wants_neuron_statistics(self):
        '''True if we want to calculate neuron segmentation accuracy'''
        return self.statistics_csv_path != "/dev/null"
    
    @property
    def wants_synapse_statistics(self):
        '''True if we are scoring synapses against ground truth'''
        return self.synapse_statistics_path != "/dev/null"
    
    @property
    def has_annotation_mask(self):
        '''True if there is a mask of the ground-truth annotated volume'''
        return self.gt_mask_channel != NO_CHANNEL
    
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
        self.synapse_classifier_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        self.additional_classifier_tasks = dict(
            [(k, np.zeros((self.n_z, self.n_y, self.n_x), object))
             for k in self.additional_neuroproof_channels])
        datasets={self.membrane_class_name: MEMBRANE_DATASET,
                  self.synapse_class_name: SYNAPSE_DATASET}
        for channel in self.additional_neuroproof_channels:
            if channel != SYNAPSE_DATASET:
                datasets[channel] = channel
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
                        datasets=datasets,
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
                    #
                    # Create a shim that returns the synapse volume
                    #
                    shim_task = ClassifyShimTask.make_shim(
                        classify_task=ctask,
                        dataset_name=SYNAPSE_DATASET)
                    self.synapse_classifier_tasks[zi, yi, xi] = shim_task
                    if SYNAPSE_DATASET in self.additional_neuroproof_channels:
                        t = self.additional_classifier_tasks[SYNAPSE_DATASET]
                        t[zi, yi, xi] = shim_task
                    for name in self.additional_neuroproof_channels:
                        shim_task = ClassifyShimTask.make_shim(
                            classify_task=ctask,
                            dataset_name=name)
                        self.additional_classifier_tasks[name][zi, yi, xi] = \
                            shim_task
    
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
                        dimensionality=self.dimensionality,
                        minimum_distance_xy=self.minimum_distance_xy,
                        minimum_distance_z=self.minimum_distance_z)
                    self.seed_tasks[zi, yi, xi] = stask
                    stask.set_requirement(ctask)

    def generate_watershed_tasks(self):
        '''Run watershed on each pixel '''
        self.watershed_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x), object)
        self.segmentation_tasks = self.watershed_tasks
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    ctask = self.classifier_tasks[zi, yi, xi]
                    btask = self.border_mask_tasks[zi, yi, xi]
                    volume = btask.volume
                    prob_target = ctask.output()
                    prob_location = DatasetLocation(
                        prob_target.paths,
                        prob_target.dataset_path,
                        prob_target.pattern)
                    seg_location = \
                        self.get_dataset_location(volume, SEG_DATASET)
                    if self.method != SeedsMethodEnum.ConnectedComponents:
                        seeds_task = self.seed_tasks[zi, yi, xi]
                        seeds_target = seeds_task.output()
                        seeds_location = seeds_target.dataset_location
                        stask = self.factory.gen_segmentation_task(
                            volume=btask.volume,
                            prob_location=prob_location,
                            mask_location=btask.mask_location,
                            seeds_location=seeds_location,
                            seg_location=seg_location,
                            sigma_xy=self.sigma_xy,
                            sigma_z=self.sigma_z,
                            dimensionality=self.dimensionality)
                        stask.set_requirement(seeds_task)
                    else:
                        stask = self.factory.gen_2D_segmentation_task(
                            volume=btask.volume,
                            prob_location=prob_location,
                            mask_location=btask.mask_location,
                            seg_location=seg_location,
                            threshold=self.threshold)
                    self.watershed_tasks[zi, yi, xi] = stask
                    stask.set_requirement(ctask)
                    stask.set_requirement(btask)
    
    def generate_resegmentation_tasks(self):
        self.resegmentation_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x), object)
        self.segmentation_tasks = self.resegmentation_tasks
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    wtask = self.watershed_tasks[zi, yi, xi]
                    input_location = wtask.output().dataset_location
                    volume = wtask.output().volume
                    output_location = self.get_dataset_location(
                        volume, RESEG_DATASET)
                    rtask = self.factory.gen_unsegmentation_task(
                        volume = volume,
                        input_location = input_location,
                        output_location = output_location,
                        use_min_contact = self.use_min_contact,
                        contact_threshold = self.contact_threshold)
                    rtask.set_requirement(wtask)
                    self.resegmentation_tasks[zi, yi, xi] = rtask
    
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
        self.x_additional_borders = \
            dict([(k, np.zeros((self.n_z, self.n_y, self.n_x-1), object))
                  for k in self.additional_neuroproof_channels])
        #
        # The task sets are composed of the input task arrays
        # the output border task arrays and the dataset name of
        # the input tasks
        #
        task_sets = [(self.classifier_tasks,
                      self.x_prob_borders,
                      MEMBRANE_DATASET),
                     (self.segmentation_tasks,
                      self.x_seg_borders,
                      SEG_DATASET)]
        if SYNAPSE_DATASET in self.additional_neuroproof_channels:
            task_sets.append((
                self.synapse_classifier_tasks,
                self.x_additional_borders[SYNAPSE_DATASET],
                SYNAPSE_DATASET))
        for i, channel in enumerate(self.additional_neuroproof_channels):
            if channel != SYNAPSE_DATASET:
                task_sets.append((
                   self.additional_classifier_tasks[channel],
                   self.x_additional_borders[channel],
                   channel))
        
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
        self.y_additional_borders = \
            dict([(k, np.zeros((self.n_z, self.n_y-1, self.n_x), object))
                  for k in self.additional_neuroproof_channels])
        #
        # The task sets are composed of the input task arrays
        # the output border task arrays and the dataset name of
        # the input tasks
        #
        task_sets = [(self.classifier_tasks,
                      self.y_prob_borders,
                      MEMBRANE_DATASET),
                     (self.segmentation_tasks,
                      self.y_seg_borders,
                      SEG_DATASET)]
        if SYNAPSE_DATASET in self.additional_neuroproof_channels:
            task_sets.append((
                self.synapse_classifier_tasks,
                self.y_additional_borders[SYNAPSE_DATASET],
                SYNAPSE_DATASET))
        for i, channel in enumerate(self.additional_neuroproof_channels):
            if channel != SYNAPSE_DATASET:
                task_sets.append((
                   self.additional_classifier_tasks[channel],
                   self.y_additional_borders[channel],
                   channel))
        
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
        self.z_additional_borders = \
            dict([(k, np.zeros((self.n_z-1, self.n_y, self.n_x), object))
                  for k in self.additional_neuroproof_channels])
        #
        # The task sets are composed of the input task arrays
        # the output border task arrays and the dataset name of
        # the input tasks
        #
        task_sets = [(self.classifier_tasks,
                      self.z_prob_borders,
                      MEMBRANE_DATASET),
                     (self.segmentation_tasks,
                      self.z_seg_borders,
                      SEG_DATASET)]
        if SYNAPSE_DATASET in self.additional_neuroproof_channels:
            task_sets.append((
                self.synapse_classifier_tasks,
                self.z_additional_borders[SYNAPSE_DATASET],
                SYNAPSE_DATASET))
        for i, channel in enumerate(self.additional_neuroproof_channels):
            if channel != SYNAPSE_DATASET:
                task_sets.append((
                   self.additional_classifier_tasks[channel],
                   self.z_additional_borders[channel],
                   channel))
        
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
        # additional classifier tasks
        # segmentation tasks
        # output tasks
        # output dataset name
        # is_border: True if neuroproofing a border, False if neuroproofing
        #            a block
        # 
        task_sets = (
            (self.classifier_tasks,
             self.additional_classifier_tasks,
             self.segmentation_tasks,
             self.np_tasks,
             NP_DATASET,
             False),
            (self.x_prob_borders,
             self.x_additional_borders,
             self.x_seg_borders,
             self.np_x_border_tasks,
             BORDER_DATASET_PATTERN.format(parent=NP_DATASET, direction="x"),
             True),
            (self.y_prob_borders,
             self.y_additional_borders,
             self.y_seg_borders,
             self.np_y_border_tasks,
             BORDER_DATASET_PATTERN.format(parent=NP_DATASET, direction="y"),
             True),
            (self.z_prob_borders,
             self.z_additional_borders,
             self.z_seg_borders,
             self.np_z_border_tasks,
             BORDER_DATASET_PATTERN.format(parent=NP_DATASET, direction="z"),
             True)
        )
        for classifier_tasks, additional_classifier_tasks, seg_tasks, np_tasks, \
            dataset_name, is_border in task_sets:

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
                        additional_tasks = [ 
                            additional_classifier_tasks[k][zi, yi, xi]
                            for k in self.additional_neuroproof_channels]
                        additional_locations = [
                            task.output().dataset_location for task in
                            additional_tasks]
                        np_task.additional_locations = additional_locations
                        np_task.set_requirement(classifier_task)
                        np_task.set_requirement(seg_task)
                        map(np_task.set_requirement, additional_tasks)
                        np_tasks[zi, yi, xi] = np_task
    
    def generate_gt_cutouts(self):
        '''Generate volumes of ground truth segmentation
        
        Get the ground-truth neuron data, the synapse data if needed and
        the mask of the annotated area, if present
        '''
        if self.wants_neuron_statistics or self.wants_synapse_statistics:
            self.gt_tasks = \
                np.zeros((self.n_z, self.n_y, self.n_x), object)
        if self.wants_synapse_statistics:
            self.gt_synapse_tasks = \
                np.zeros((self.n_z, self.n_y, self.n_x), object)
        if self.has_annotation_mask:
            self.gt_mask_tasks = \
                np.zeros((self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            z0 = self.zs[zi]
            z1 = self.zs[zi+1]
            for yi in range(self.n_y):
                y0 = self.ys[yi]
                y1 = self.ys[yi+1]
                for xi in range(self.n_x):
                    x0 = self.xs[xi]
                    x1 = self.xs[xi+1]
                    volume=Volume(x0, y0, z0, x1-x0, y1-y0, z1-z0)
                    self.generate_gt_cutout(volume, yi, xi, zi)

    def generate_gt_cutout(self, volume, yi, xi, zi):
        '''Generate gt cutouts for a given volume'''
        dataset_location = self.get_dataset_location(
            volume, GT_DATASET)
        if self.wants_neuron_statistics or self.wants_synapse_statistics:
            btask = self.factory.gen_get_volume_task(
                self.experiment,
                self.sample,
                self.dataset,
                self.gt_channel,
                self.url,
                volume,
                dataset_location)
            self.gt_tasks[zi, yi, xi] = btask
        if self.wants_synapse_statistics:
            synapse_gt_location = self.get_dataset_location(
                volume, SYN_GT_DATASET)
            self.gt_synapse_tasks[zi, yi, xi] =\
                self.factory.gen_get_volume_task(
                    experiment=self.experiment,
                    sample=self.sample,
                    dataset=self.dataset,
                    channel=self.synapse_channel,
                    url=self.url,
                    volume=volume,
                    location=synapse_gt_location)
        if self.has_annotation_mask:
            gt_mask_location = self.get_dataset_location(
                volume, GT_MASK_DATASET)
            self.gt_mask_tasks[zi, yi, xi] = \
                self.factory.gen_get_volume_task(
                    experiment=self.experiment,
                    sample=self.sample,
                    dataset=self.dataset,
                    channel=self.gt_mask_channel,
                    url=self.url,
                    volume=volume,
                    location=gt_mask_location)
            
    def generate_pred_cutouts(self):
        '''Generate volumes matching the ground truth segmentations'''
        self.pred_block_tasks = np.zeros((self.n_z, self.n_y, self.n_x),
                                         object)
        self.gt_block_tasks = np.zeros((self.n_z, self.n_y, self.n_x),
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
                    #
                    # Reblock the segmentation prediction
                    #
                    dataset_location = self.get_dataset_location(
                        volume, PRED_DATASET)
                    nptask = self.np_tasks[zi, yi, xi]
                    btask = self.factory.gen_block_task(
                        volume, dataset_location,
                        [dict(volume=nptask.volume,
                              location=nptask.output_seg_location)])
                    btask.set_requirement(nptask)
                    self.pred_block_tasks[zi, yi, xi] = btask
                    #
                    # Reblock the ground-truth
                    #
                    dataset_location = self.get_dataset_location(
                        volume, GT_BLOCK_DATASET)
                    gt_task = self.gt_tasks[zi, yi, xi]
                    btask = self.factory.gen_block_task(
                        volume, dataset_location,
                        [dict(volume=gt_task.volume,
                              location=gt_task.output().dataset_location)])
                    btask.set_requirement(gt_task)
                    self.gt_block_tasks[zi, yi, xi] = btask
                    
    def generate_statistics_tasks(self):
        if self.wants_neuron_statistics:
            self.statistics_tasks = np.zeros((self.n_z, self.n_y, self.n_x),
                                             object)
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
                            gt_seg_location=gttask.output().dataset_location,
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
                output_path = self.statistics_csv_path,
                excluded_keys=["per_object"])
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
    
    def generate_skeletonize_tasks(self):
        '''Generate tasks that skeletonize the major Neuroproofed segmentations
        
        '''
        if self.wants_skeletonization:
            self.skeletonize_tasks = np.zeros((self.n_z, self.n_y, self.n_x),
                                              object)
            for zi in range(self.n_z):
                for yi in range(self.n_y):
                    for xi in range(self.n_x):
                        ntask = self.np_tasks[zi, yi, xi]
                        volume = ntask.volume
                        seg_location = ntask.output_seg_location
                        skel_root = self.get_dirs(self.xs[xi],
                                                  self.ys[yi],
                                                  self.zs[zi])
                        skel_location = os.path.join(
                            skel_root[0], SKEL_DIR_NAME)
                        stask = self.factory.gen_skeletonize_task(
                            volume=volume,
                            segmentation_location=seg_location,
                            skeleton_location=skel_location)
                        stask.set_requirement(ntask)
                        self.skeletonize_tasks[zi, yi, xi] = stask
    
    def generate_connectivity_graph_tasks(self):
        '''Create the tasks that join components across blocks'''
        if not self.wants_connectivity:
            return
        #
        # Find the connected components between the big blocks and the
        # border blocks. Then knit them together with the 
        # AllConnectedComponentsTask
        #
        self.generate_x_connectivity_graph_tasks()
        self.generate_y_connectivity_graph_tasks()
        self.generate_z_connectivity_graph_tasks()
        input_tasks = []
        for task_array in (self.x_connectivity_graph_tasks,
                           self.y_connectivity_graph_tasks,
                           self.z_connectivity_graph_tasks):
            input_tasks += task_array.flatten().tolist()
        #
        # Apply parameterizations common to x, y and z
        #
        for task in input_tasks:
            task.min_overlap_percent = self.min_percent_connected
        #
        # Build the all-connected-components task
        #
        input_locations = [task.output().path for task in input_tasks]
        self.all_connected_components_task = \
            self.factory.gen_all_connected_components_task(
                input_locations, self.connectivity_graph_location)
        for task in input_tasks:
            self.all_connected_components_task.set_requirement(task)
    
    def generate_x_connectivity_graph_tasks(self):
        '''Generate connected components tasks to link blocks in x direction
        
        '''
        self.x_connectivity_graph_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x-1, 2), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x-1):
                    border_task = self.np_x_border_tasks[zi, yi, xi]
                    border_volume = border_task.volume
                    left_task = self.np_tasks[zi, yi, xi]
                    right_task = self.np_tasks[zi, yi, xi+1]
                    #
                    # The left task overlap is 1/2 of the padding away
                    # from its right edge and 1/2 of the padding (= 1/4 of
                    # the width) from the left edge of the border block
                    #
                    overlap_volume1 = Volume(
                        border_volume.x + self.np_x_pad / 2,
                        border_volume.y,
                        border_volume.z,
                        1, 
                        border_volume.height,
                        border_volume.depth)
                    #
                    # The right task overlap is 1/2 of the padding away
                    # from its left edge and 1/2 of the padding (= 3/4 of
                    # the width) from the right edge of the border block
                    #
                    overlap_volume2 = Volume(
                        self.xs[xi+1] + self.np_x_pad / 2,
                        border_volume.y,
                        border_volume.z,
                        1, 
                        border_volume.height, 
                        border_volume.depth)
                    for idx, np_task, volume, direction in (
                        (0, left_task, overlap_volume1, "x-"),
                        (1, right_task, overlap_volume2, "x+")):
                        filename = CONNECTED_COMPONENTS_PATTERN.format(
                            direction=direction)
                        output_location = os.path.join(
                            np_task.output_seg_location.roots[0], filename)
                        task = self.factory.gen_connected_components_task(
                            volume1=np_task.volume,
                            location1=np_task.output_seg_location,
                            volume2=border_volume,
                            location2=border_task.output_seg_location,
                            overlap_volume=volume,
                            output_location=output_location)
                        task.set_requirement(np_task)
                        task.set_requirement(border_task)
                        self.x_connectivity_graph_tasks[zi, yi, xi, idx] = task
                                        
    def generate_y_connectivity_graph_tasks(self):
        '''Generate connected components tasks to link blocks in y direction
        
        '''
        self.y_connectivity_graph_tasks = np.zeros(
            (self.n_z, self.n_y-1, self.n_x, 2), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y - 1):
                for xi in range(self.n_x):
                    border_task = self.np_y_border_tasks[zi, yi, xi]
                    border_volume = border_task.volume
                    left_task = self.np_tasks[zi, yi, xi]
                    right_task = self.np_tasks[zi, yi+1, xi]
                    overlap_volume1 = Volume(
                        border_volume.x,
                        border_volume.y + self.np_y_pad / 2,
                        border_volume.z,
                        border_volume.width, 
                        1, 
                        border_volume.depth)
                    overlap_volume2 = Volume(
                        border_volume.x,
                        self.ys[yi+1] + self.np_y_pad / 2,
                        border_volume.z,
                        border_volume.width, 
                        1, 
                        border_volume.depth)
                    for idx, np_task, volume, direction in (
                        (0, left_task, overlap_volume1, "y-"),
                        (1, right_task, overlap_volume2, "y+")):
                        filename = CONNECTED_COMPONENTS_PATTERN.format(
                            direction=direction)
                        output_location = os.path.join(
                            np_task.output_seg_location.roots[0], filename)
                        task = self.factory.gen_connected_components_task(
                            volume1=np_task.volume,
                            location1=np_task.output_seg_location,
                            volume2=border_volume,
                            location2=border_task.output_seg_location,
                            overlap_volume=volume,
                            output_location=output_location)
                        task.set_requirement(np_task)
                        task.set_requirement(border_task)
                        self.y_connectivity_graph_tasks[zi, yi, xi, idx] = task
                                        
    def generate_z_connectivity_graph_tasks(self):
        '''Generate connected components tasks to link blocks in z direction
        
        '''
        self.z_connectivity_graph_tasks = np.zeros(
            (self.n_z-1, self.n_y, self.n_x, 2), object)
        for zi in range(self.n_z-1):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    border_task = self.np_z_border_tasks[zi, yi, xi]
                    border_volume = border_task.volume
                    left_task = self.np_tasks[zi, yi, xi]
                    right_task = self.np_tasks[zi+1, yi, xi]
                    overlap_volume1 = Volume(
                        border_task.volume.x,
                        border_task.volume.y,
                        border_task.volume.z + self.np_z_pad / 2,
                        border_task.volume.width, 
                        border_task.volume.height, 1)
                    overlap_volume2 = Volume(
                        border_task.volume.x,
                        border_task.volume.y,
                        self.zs[zi+1] + self.np_z_pad / 2,
                        border_task.volume.width, 
                        border_task.volume.height, 1)
                    for idx, np_task, volume, direction in (
                        (0, left_task, overlap_volume1, "z-"),
                        (1, right_task, overlap_volume2, "z+")):
                        filename = CONNECTED_COMPONENTS_PATTERN.format(
                            direction=direction)
                        output_location = os.path.join(
                            np_task.output_seg_location.roots[0], filename)
                        task = self.factory.gen_connected_components_task(
                            volume1=np_task.volume,
                            location1=np_task.output_seg_location,
                            volume2=border_volume,
                            location2=border_task.output_seg_location,
                            overlap_volume=volume,
                            output_location=output_location)
                        task.set_requirement(np_task)
                        task.set_requirement(border_task)
                        self.z_connectivity_graph_tasks[zi, yi, xi, idx] = task
    
    def generate_synapse_segmentation_tasks(self):
        '''Generate connected-components and filter tasks for synapses
        
        '''
        #
        # Segment the synapses using connected components on a blurred
        # probability map, then filter by size
        #
        self.synapse_segmentation_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    ctask = self.synapse_classifier_tasks[zi, yi, xi]
                    nptask = self.np_tasks[zi, yi, xi]
                    volume = ctask.output().volume
                    ctask_loc = ctask.output().dataset_location
                    np_loc = nptask.output().dataset_location
                    stask_loc = self.get_dataset_location(
                        volume, SYN_SEG_DATASET)
                    stask = self.factory.gen_find_synapses_task(
                        volume=volume,
                        syn_location=ctask_loc,
                        neuron_segmentation=np_loc,
                        output_location=stask_loc,
                        threshold=self.synapse_threshold,
                        erosion_xy=self.synapse_xy_erosion,
                        erosion_z=self.synapse_z_erosion,
                        sigma_xy=self.synapse_xy_sigma,
                        sigma_z=self.synapse_z_sigma,
                        min_size_2d=self.synapse_min_size_2d,
                        max_size_2d=self.synapse_max_size_2d,
                        min_size_3d=self.min_synapse_area,
                        min_slice=self.min_synapse_depth)
                    stask.set_requirement(ctask)
                    stask.set_requirement(nptask)
                    self.synapse_segmentation_tasks[zi, yi, xi] = stask
    
    def generate_synapse_connectivity_tasks(self):
        '''Make tasks that connect neurons to synapses'''
        self.synapse_connectivity_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    segtask = self.synapse_segmentation_tasks[zi, yi, xi]
                    volume = segtask.output().volume
                    syn_location = segtask.output().dataset_location
                    ntask = self.np_tasks[zi, yi, xi]
                    neuron_location = ntask.output().dataset_location
                    output_location = os.path.join(
                        self.get_dirs(
                            self.xs[xi], self.ys[yi], self.zs[zi])[0],
                        "synapse_connectivity.json")
                    
                    sctask = self.factory.gen_connect_synapses_task(
                        volume=volume,
                        synapse_location=syn_location,
                        neuron_location=neuron_location,
                        output_location=output_location,
                        xy_dilation=self.synapse_xy_dilation,
                        z_dilation=self.synapse_z_dilation,
                        min_contact=self.min_synapse_neuron_contact)
                    sctask.set_requirement(segtask)
                    sctask.set_requirement(ntask)
                    self.synapse_connectivity_tasks[zi, yi, xi] = sctask
    
    def generate_synapse_statistics_tasks(self):
        '''Make tasks that calculate precision/recall on synapses'''
        if not self.wants_synapse_statistics:
            return
        #
        # This is a pipeline in and of itself:
        #
        # Butterfly -> synapse ground truth
        # Synapse ground truth -> segmentation
        # Connect GT neurites to GT synapses
        # Match GT synapses against detected synapses
        # Match GT neurites against detected neurites
        # Compile a global mapping of synapse to neuron
        # Calculate statistics based on this information
        # 
        synapse_neuron_connection_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        d_gt_neuron_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        d_gt_synapse_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            z0 = self.zs[zi]
            z1 = self.zs[zi+1]
            for yi in range(self.n_y):
                y0 = self.ys[yi]
                y1 = self.ys[yi+1]
                for xi in range(self.n_x):
                    x0 = self.xs[xi]
                    x1 = self.xs[xi+1]
                    volume=Volume(x0, y0, z0, x1-x0, y1-y0, z1-z0)
                    synapse_gt_task = self.gt_synapse_tasks[zi, yi, xi]
                    synapse_gt_location = \
                        synapse_gt_task.output().dataset_location
                    #
                    # Segment
                    #
                    synapse_gt_seg_location = self.get_dataset_location(
                        volume, SYN_SEG_GT_DATASET)
                    synapse_gt_seg_task = self.factory.gen_cc_segmentation_task(
                        volume=volume,
                        prob_location=synapse_gt_seg_location,
                        mask_location=EMPTY_DATASET_LOCATION,
                        seg_location = synapse_gt_seg_location,
                        threshold=0,
                        dimensionality=Dimensionality.D3,
                        fg_is_higher=True)
                    synapse_gt_seg_task.set_requirement(synapse_gt_task)
                    #
                    # Connect gt segments to gt neurites
                    #
                    gt_neuron_task = self.gt_tasks[zi, yi, xi]
                    gt_synapse_neuron_location = os.path.join(
                        self.get_dirs(x0, y0, z0)[0], 
                        "gt-synapse-neuron-connections.json")
                    gt_synapse_neuron_task = \
                        self.factory.gen_connected_components_task(
                            volume,
                            gt_neuron_task.output().dataset_location,
                            volume,
                            synapse_gt_seg_location,
                            volume,
                            gt_synapse_neuron_location)
                    gt_synapse_neuron_task.set_requirement(gt_neuron_task)
                    gt_synapse_neuron_task.set_requirement(synapse_gt_seg_task)
                    synapse_neuron_connection_tasks[zi, yi, xi] = \
                        gt_synapse_neuron_task
                    #
                    # Match GT synapses against detected synapses
                    #
                    synapse_seg_task = \
                        self.synapse_segmentation_tasks[zi, yi, xi]
                    synapse_seg_location = \
                        synapse_seg_task.output().dataset_location
                    synapse_match_location = os.path.join(
                        self.get_dirs(x0, y0, z0)[0], "synapse-match.json")
                    synapse_match_task = self.factory.gen_match_synapses_task(
                        volume=volume,
                        gt_location=synapse_gt_seg_location,
                        detected_location=synapse_seg_location,
                        output_location=synapse_match_location,
                        method=self.synapse_match_method)
                    synapse_match_task.set_requirement(synapse_seg_task)
                    synapse_match_task.set_requirement(synapse_gt_seg_task)
                    if self.has_annotation_mask:
                        gt_mask_task = self.gt_mask_tasks[zi, yi, xi]
                        gt_mask_location = \
                            gt_mask_task.output().dataset_location
                        synapse_match_task.mask_location = gt_mask_location
                        synapse_match_task.set_requirement(gt_mask_task)
                    d_gt_synapse_tasks[zi, yi, xi] = synapse_match_task
                    #
                    # Match GT neurons against detected neurons
                    #
                    neuron_match_location = os.path.join(
                        self.get_dirs(x0, y0, z0)[0], "neuron-match.json")
                    neuron_seg_task = self.segmentation_tasks[zi, yi, xi]
                    neuron_seg_location = \
                        neuron_seg_task.output().dataset_location
                    neuron_match_task=self.factory.gen_match_neurons_task(
                        volume=volume,
                        gt_location=gt_neuron_task.output().dataset_location,
                        detected_location=neuron_seg_location,
                        output_location=neuron_match_location)
                    neuron_match_task.set_requirement(neuron_seg_task)
                    neuron_match_task.set_requirement(gt_neuron_task)
                    d_gt_neuron_tasks[zi, yi, xi] = neuron_match_task
        #
        # Compile the global synapse / neuron connection map
        #
        synapse_neuron_connections = [
            task.output().path 
            for task in synapse_neuron_connection_tasks.flatten()]
        synapse_gt_location = os.path.join(
            self.temp_dirs[0],"synapse-gt-neuron-connections.json")
        synapse_gt_task = SynapseGtTask(
            synapse_neuron_connection_locations=synapse_neuron_connections,
            output_location=synapse_gt_location)
        map(synapse_gt_task.set_requirement, 
            synapse_neuron_connection_tasks.flatten())
        #
        # Create the statistics task
        #
        def locs_of(tasks):
            return [task.output().path for task in tasks]
        statistics_task_location = "synapse-statistics.json"
        synapse_match_tasks = d_gt_synapse_tasks.flatten()
        detected_synapse_connection_tasks = \
            synapse_neuron_connection_tasks.flatten()
        gt_neuron_map_tasks = d_gt_neuron_tasks.flatten()
        self.synapse_statistics_task = self.factory.gen_synapse_statistics_task(
            locs_of(synapse_match_tasks),
            locs_of(synapse_neuron_connection_tasks.flatten()),
            neuron_map=self.all_connected_components_task.output().path,
            gt_neuron_maps=locs_of(gt_neuron_map_tasks),
            gt_synapse_connections=synapse_gt_location,
            output_location=self.synapse_statistics_path)
        self.synapse_statistics_task.set_requirement(
            self.all_connected_components_task)
        self.synapse_statistics_task.set_requirement(synapse_gt_task)
        map(self.synapse_statistics_task.set_requirement, 
            synapse_match_tasks)
        map(self.synapse_statistics_task.set_requirement,
            synapse_neuron_connection_tasks)
        map(self.synapse_statistics_task.set_requirement,
            gt_neuron_map_tasks)
        
    def compute_requirements(self):
        '''Compute the requirements for this task'''
        if not hasattr(self, "requirements"):
            try:
                rh_logger.logger.report_event("Assembling pipeline")
            except:
                rh_logger.logger.start_process("Ariadne pipeline",
                                               "Assembling pipeline")
                #
                # Configuration turns off the luigi-interface logger
                #
            import logging
            logging.getLogger("luigi-interface").disabled = False
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
                rh_logger.logger.report_event("Making gt cutout tasks")
                self.generate_gt_cutouts()
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
                if self.method != SeedsMethodEnum.ConnectedComponents:
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
                if self.wants_resegmentation:
                    self.generate_resegmentation_tasks()
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
                # Step 8: Skeletonize Neuroproof
                #
                rh_logger.logger.report_event("Making skeletonize tasks")
                self.generate_skeletonize_tasks()
                #
                # Step 9: Segment the synapses
                #
                rh_logger.logger.report_event("Segment synapses")
                self.generate_synapse_segmentation_tasks()
                #
                # Step 10: The connectivity graph.
                #
                rh_logger.logger.report_event("Making connectivity graph")
                self.generate_connectivity_graph_tasks()
                #
                # Step 11: Connect synapses to neurites
                #
                rh_logger.logger.report_event("Connecting synapses and neurons")
                self.generate_synapse_connectivity_tasks()
                #
                # Step 12: find ground-truth synapses and compute statistics
                #
                rh_logger.logger.report_event("Comparing synapses to gt")
                self.generate_synapse_statistics_tasks()
                #
                # The requirements:
                #
                # The skeletonize tasks if skeletonization is done
                #     otherwise the block neuroproof tasks
                # The border neuroproof tasks
                # The statistics task
                #
                self.requirements = []
                if self.wants_skeletonization:
                    self.requirements += \
                        self.skeletonize_tasks.flatten().tolist()
                else:
                    self.requirements += self.np_tasks.flatten().tolist()
                if self.wants_connectivity:
                    self.requirements.append(self.all_connected_components_task)
                if self.wants_synapse_statistics:
                    self.requirements.append(self.synapse_statistics_task)
                self.requirements += \
                    self.synapse_connectivity_tasks.flatten().tolist()
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
    
        
        
        