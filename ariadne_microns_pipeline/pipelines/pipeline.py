import luigi
import rh_logger
from .utilities import PipelineRunReportMixin
from ..tasks.factory import AMTaskFactory
from ..tasks.classify import ClassifyShimTask
from ..tasks.connected_components import JoiningMethod
from ..tasks.connected_components import FakeAllConnectedComponentsTask
from ..tasks.find_seeds import SeedsMethodEnum, Dimensionality
from ..tasks.match_synapses import MatchMethod
from ..tasks.nplearn import StrategyEnum
from ..targets.classifier_target import PixelClassifierTarget
from ..targets.hdf5_target import HDF5FileTarget
from ..targets.butterfly_target import ButterflyChannelTarget
from ..parameters import Volume, VolumeParameter, DatasetLocation
from ..parameters import EMPTY_DATASET_LOCATION, is_empty_dataset_location
from ..tasks.utilities import to_hashable
from ..pipelines.synapse_gt_pipeline import SynapseGtTask
import json
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

'''The name of the X affinity probability datasets'''
X_AFFINITY_DATASET = "x-affinity"

'''The name of the Y affinity probability datasets'''
Y_AFFINITY_DATASET = "y-affinity"

'''The name of the Z affinity probability datasets'''
Z_AFFINITY_DATASET = "z-affinity"

'''The name of the synapse probability datasets'''
SYNAPSE_DATASET = "synapse"

'''The name of the synapse transmitter probability datasets'''
SYNAPSE_TRANSMITTER_DATASET = "transmitter"

'''The name of the synapse receptor probability datasets'''
SYNAPSE_RECEPTOR_DATASET = "receptor"

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

'''The name of the final stitched segmentation'''
FINAL_SEGMENTATION = "final-segmentation"

'''The pattern for border datasets

parent - name of parent dataset, e.g. "membrane"
direction - the adjacency direction, e.g. "z"
'''
BORDER_DATASET_PATTERN = "{parent}_{direction}-border"

'''The pattern for connected_components tasks

direction - the adjacency direction: x-, x+, y-, y+, z-, z+
'''
CONNECTED_COMPONENTS_PATTERN = "connected-components_{direction}.json"

'''The name of the connected components JSON file if not specified'''
ALL_CONNECTED_COMPONENTS_JSON = "connected-components.json"

'''Signals that the channel isn't available (e.g. no ground truth)'''
NO_CHANNEL = "no-channel"

'''Prepended to stitched versions of dataset names for nplearn'''
NPLEARN_PREFIX = "nplearn-"

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
    xy_nm = luigi.FloatParameter(
        default=4.0,
        description="The size of a voxel in the x and y directions")
    z_nm = luigi.FloatParameter(
        default=30.0,
        description="The size of a voxel in the z direction")
    #########
    #
    # Optional parameters
    #
    #########
    resolution = luigi.IntParameter(
        default=0,
        description="The MIPMAP resolution of the volume to be processed.")
    block_width = luigi.IntParameter(
        description="Width of one of the processing blocks",
        default=2048)
    block_height = luigi.IntParameter(
        description="Height of one of the processing blocks",
        default=2048)
    block_depth = luigi.IntParameter(
        description="Number of planes in a processing block",
        default=50)
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
    np_threshold = luigi.FloatParameter(
        default=.2,
        description="The probability threshold for merging in Neuroproof "
        "(range = 0-1).")
    np_cores = luigi.IntParameter(
        description="The number of cores used by a Neuroproof process",
        default=2)
    temp_dirs = luigi.ListParameter(
        description="The base location for intermediate files",
        default=(tempfile.gettempdir(),))
    membrane_class_name = luigi.Parameter(
        description="The name of the pixel classifier's membrane class",
        default="membrane")
    wants_affinity_segmentation = luigi.BoolParameter(
        description="Use affinity probabilities and z-watershed to "
        "segment the volume.")
    x_affinity_class_name = luigi.Parameter(
        default="x",
        description="The name of the affinity classifier's class connecting "
        "voxels in the x direction.")
    y_affinity_class_name = luigi.Parameter(
        default="y",
        description="The name of the affinity classifier's class connecting "
        "voxels in the y direction.")
    z_affinity_class_name = luigi.Parameter(
        default="z",
        description="The name of the affinity classifier's class connecting "
        "voxels in the z direction.")
    z_watershed_threshold = luigi.IntParameter(
        default=40000,
        description="Target size for segments produced by the z-watershed")
    wants_transmitter_receptor_synapse_maps = luigi.BoolParameter(
        description="Use a synapse transmitter and receptor probability map "
                    "instead of a map of synapse voxel probabilities.")
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
    wants_neuroproof_learn = luigi.BoolParameter(
        description="Train Neuroproof's classifier")
    additional_neuroproof_channels = luigi.ListParameter(
        default=[],
        description="The names of additional classifier classes "
                    "that are fed into Neuroproof as channels")
    wants_standard_neuroproof=luigi.BoolParameter(
        description="Use the standard Neuroproof build and interface")
    nplearn_strategy = luigi.EnumParameter(
        enum=StrategyEnum,
        default=StrategyEnum.all,
        description="Learning strategy to use")
    nplearn_num_iterations = luigi.IntParameter(
        default=1,
        description="# of iterations of Neuroproof learning to run")
    prune_feature = luigi.BoolParameter(
        description="Automatically prune useless features")
    use_mito = luigi.BoolParameter(
        description="Set delayed mito agglomeration")
    nplearn_cpu_count = luigi.IntParameter(
        default=4,
        description="# of CPUS to use in the NeuroproofLearnTask")
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
    watershed_threshold = luigi.IntParameter(
        default=192,
        description="The threshold to use when computing the distance from "
        "membrane")
    use_distance_watershed = luigi.BoolParameter(
        description="Use the distance transform when computing watershed")
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
    skeleton_decimation_factor = luigi.FloatParameter(
        description="Remove skeleton leaves if they are less than this factor "
        "of their parent's volume",
        default=.5)
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
    
    stitched_segmentation_location = luigi.Parameter(
        default="/dev/null",
        description="The location for the final stitched segmentation")
    index_file_location = luigi.Parameter(
        default="/dev/null",
        description="A JSON file that maps volumes to datasets")
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
    synapse_connection_location = luigi.Parameter(
        default="/dev/null",
        description="The location for the JSON file containing the global "
                    "IDs of the neuron partners for each synapse and the "
                    "coordinates of that synapse.")
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
    gt_neuron_synapse_xy_dilation = luigi.IntParameter(
        default=6,
        description="The number of pixels of dilation to apply to the synapses "
                    "in the ground-truth dataset in the x and y directions "
                    "before matching to neurons")
    gt_neuron_synapse_z_dilation = luigi.IntParameter(
        default=0,
        description="The number of pixels of dilation to apply to the synapses "
                    "in the ground-truth dataset in the z direction "
                    "before matching to neurons")
    gt_neuron_synapse_min_contact = luigi.IntParameter(
        default=0,
        description="The minimum amount of contact between a ground-truth "
                    "neuron and syapse before they can be considered "
                    "to be touching.")
    #
    # Parameters for block joining
    #
    joining_method = luigi.EnumParameter(
        enum=JoiningMethod,
        default=JoiningMethod.PAIRWISE_MULTIMATCH,
        description="Algorithm to use to join neuroproofed segmentation blocks")
    min_percent_connected = luigi.FloatParameter(
        default=75.0,
        description="Minimum overlap required to join segments across blocks")
    min_overlap_volume = luigi.IntParameter(
        default=1000,
        description="The minimum # of voxels of overlap between two objects "
                    "required to join them across blocks")
    max_poly_matches = luigi.IntParameter(
        default=1)
    dont_join_orphans = luigi.BoolParameter()
    orphan_min_overlap_ratio = luigi.FloatParameter(
        default=0.9)
    orphan_min_overlap_volume = luigi.IntParameter(
        default=1000,
        description="The minimum # of voxels of overlap needed to join "
                    "an orphan segment")
    halo_size_xy = luigi.IntParameter(
        default=5,
        description="The number of pixels on either side of the origin to "
                    "use as context when extracting the slice to be joined, "
                    "joining slices in the x and y directions")
    halo_size_z = luigi.IntParameter(
        default=1,
        description="The number of pixels on either side of the origin to "
                    "use as context when extracting the slice to be joined, "
                    "joining slices in the z direction")

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
        return self.connectivity_graph_location != "/dev/null" or \
            self.stitched_segmentation_location != "/dev/null" or \
            self.wants_neuron_statistics or \
            self.wants_synapse_statistics or \
            self.wants_neuroproof_learn
    
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
        
        This routine computes the extents as follows:
        
        * Blocks are always the stated block size (unless the block size is
          less than the entire volume).
        
        * For classification, blocks are extended by the classifier padding
          when fetching image data. It's assumed that there is valid image
          data of at least the padding size on every side of the volume.
          Classifier blocks must be at least the block size but can be
          larger.
        
        * A reblocking is done after the classification. These blocks have
          at least the required amount of Neuroproof overlap padding with
          adjacent blocks. The block size is maintained which may result
          in an excess of padding - this assumes that more context for
          Neuroproof is a good thing and border regions are a bad thing.
        
        self.{x,y,z}{0,1} - the start and end coordinates of the volume to be
                            analyzed. This might be smaller than the requested
                            volume if the Butterfly volume is smaller than
                            the requested volume.
        self.nn_{x,y,z}_pad - amount of padding for pixel classifier ("nn"
                              stands for Neural Network).
        
        self.cl_{x, y, z}{s,e} - the start and end extents in the x, y & z dirs
                                 for the classifier blocks, before padding.
        
        self.cl_padded_{x, y, z} - the start and end extents in the x, y & z
                                   directions for the classifier blocks
                                   after padding.

        self.ncl_{x, y, z} - the number of blocks in the x, y and z dirs
                             for the classifiers.

        self.{x, y, z}{s,e} - the starts and ends of each block
        
        self.{x,y,z}_grid - the grid for the valid sections of each block.
                            For instance, block xi's valid region, considering
                            overlap is self.x_grid[xi] to self.x_grid[xi+1]
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
        # Compute exact block sizes for the classifier w/o overlap
        #
        self.ncl_x = int((self.useable_width-1) / self.block_width) + 1
        self.ncl_y = int((self.useable_height-1) / self.block_height) + 1
        self.ncl_z = int((self.useable_depth-1) / self.block_depth) + 1
        self.cl_xs = self.x0 + self.block_width * np.arange(self.ncl_x)
        self.cl_xe = self.cl_xs + self.block_width
        self.cl_ys = self.y0 + self.block_height * np.arange(self.ncl_y)
        self.cl_ye = self.cl_ys + self.block_width
        self.cl_zs = self.z0 + self.block_depth * np.arange(self.ncl_z)
        self.cl_ze = self.cl_zs + self.block_depth
        #
        # Compute # of blocks for segmentation and beyond. We need at least
        # the Neuroproof padding between n-1 blocks.
        #
        self.n_x = int((self.useable_width - self.block_width - 1) /
                       (self.block_width - self.np_x_pad)) + 2
        self.n_y = int((self.useable_height - self.block_height - 1) /
                       (self.block_height - self.np_y_pad)) + 2
        self.n_z = int((self.useable_depth - self.block_depth - 1) /
                       (self.block_depth - self.np_z_pad)) + 2
        self.xs = np.linspace(self.x0, self.x1, self.n_x, endpoint = False)\
            .astype(int)
        self.xe = np.minimum(self.xs + self.block_width, self.x1)
        self.ys = np.linspace(self.y0, self.y1, self.n_y, endpoint = False)\
            .astype(int)
        self.ye = np.minimum(self.ys + self.block_height, self.y1)
        self.zs = np.linspace(self.z0, self.z1, self.n_z, endpoint=False)\
            .astype(int)
        self.ze = np.minimum(self.zs + self.block_depth, self.z1)
        #
        # The first and last valid blocks start and end at the extents.
        # The intermediate blocks start and end midway between the overlap
        # between.
        #
        self.x_grid = np.hstack(([self.x0], 
                                 (self.xs[1:] + self.xe[:-1]) / 2,
                                 [self.x1])).astype(int)
        self.y_grid = np.hstack(([self.y0], 
                                 (self.ys[1:] + self.ye[:-1]) / 2,
                                 [self.y1])).astype(int)
        self.z_grid = np.hstack(([self.z0], 
                                 (self.zs[1:] + self.ze[:-1]) / 2,
                                 [self.z1])).astype(int)

    def generate_butterfly_tasks(self):
        '''Get volumes padded for CNN'''
        self.butterfly_tasks = \
            np.zeros((self.ncl_z, self.ncl_y, self.ncl_x), object)
        for zi in range(self.ncl_z):
            z0 = self.cl_zs[zi] - self.nn_z_pad
            z1 = self.cl_ze[zi] + self.nn_z_pad
            for yi in range(self.ncl_y):
                y0 = self.cl_ys[yi] - self.nn_y_pad
                y1 = self.cl_ye[yi] + self.nn_y_pad
                for xi in range(self.ncl_x):
                    x0 = self.cl_xs[xi] - self.nn_x_pad
                    x1 = self.cl_xe[xi] + self.nn_x_pad
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
                            location=location,
                            resolution=self.resolution)
                    self.register_dataset(
                        self.butterfly_tasks[zi, yi, xi].output())

    def generate_classifier_tasks(self):
        '''Get the pixel classifier tasks
        
        Take each butterfly task and run a pixel classifier on its output.
        '''
        #
        # The datasets dictionary maps a class name baked into the classifier
        # to a generic name, for instance "membrane".
        #
        # The taskmaps dictionary maps a generic name to the 3D array that
        # stores the shim task associated with that generic name.
        #
        if self.wants_affinity_segmentation:
            self.classifier_tasks = \
                np.zeros((3, self.ncl_z, self.ncl_y, self.ncl_x), object)
            datasets = {
                self.x_affinity_class_name:X_AFFINITY_DATASET,
                self.y_affinity_class_name:Y_AFFINITY_DATASET,
                self.z_affinity_class_name:Z_AFFINITY_DATASET
                }
            taskmaps = {
                Z_AFFINITY_DATASET: self.classifier_tasks[0],
                Y_AFFINITY_DATASET: self.classifier_tasks[1],
                X_AFFINITY_DATASET: self.classifier_tasks[2]
            }
        else:
            self.classifier_tasks = \
                np.zeros((self.ncl_z, self.ncl_y, self.ncl_x), object)
            datasets = {
                self.membrane_class_name: MEMBRANE_DATASET
            }
            taskmaps = {
                MEMBRANE_DATASET: self.classifier_tasks
            }
            
        if not self.wants_neuroproof_learn:
            if self.wants_transmitter_receptor_synapse_maps:
                self.transmitter_classifier_tasks = \
                    np.zeros((self.ncl_z, self.ncl_y, self.ncl_x), object)
                self.receptor_classifier_tasks = \
                    np.zeros((self.ncl_z, self.ncl_y, self.ncl_x), object)
                datasets.update({
                    self.transmitter_class_name: SYNAPSE_TRANSMITTER_DATASET, 
                    self.receptor_class_name: SYNAPSE_RECEPTOR_DATASET
                })
                taskmaps.update({
                    SYNAPSE_TRANSMITTER_DATASET: self.transmitter_classifier_tasks,
                    SYNAPSE_RECEPTOR_DATASET: self.receptor_classifier_tasks
                })
            else:
                self.synapse_classifier_tasks = np.zeros(
                    (self.ncl_z, self.ncl_y, self.ncl_x), object)
                datasets.update({
                    self.synapse_class_name: SYNAPSE_DATASET})
                taskmaps.update({
                    SYNAPSE_DATASET: self.synapse_classifier_tasks
                })
        self.additional_classifier_tasks = dict(
            [(k, np.zeros((self.ncl_z, self.ncl_y, self.ncl_x), object))
             for k in self.additional_neuroproof_channels])
        for channel in self.additional_neuroproof_channels:
            if channel not in (SYNAPSE_DATASET, SYNAPSE_TRANSMITTER_DATASET,
                               SYNAPSE_RECEPTOR_DATASET) \
               and not self.wants_neuroproof_learn:
                datasets[channel] = self.additional_classifier_tasks[channel]
        for zi in range(self.ncl_z):
            for yi in range(self.ncl_y):
                for xi in range(self.ncl_x):
                    btask = self.butterfly_tasks[zi, yi, xi]
                    input_target = btask.output()
                    img_location = DatasetLocation(
                        input_target.paths,
                        input_target.dataset_path,
                        input_target.pattern)
                    paths = self.get_dirs(
                        self.cl_xs[xi], self.cl_ys[yi], self.cl_zs[zi])
                    ctask = self.factory.gen_classify_task(
                        paths=paths,
                        datasets=datasets,
                        pattern=self.get_pattern(MEMBRANE_DATASET),
                        img_volume=btask.volume,
                        img_location=img_location,
                        classifier_path=self.pixel_classifier_path)
                    ctask.set_requirement(btask)
                    #
                    # Create shims for all channels
                    #
                    for channel, tasks in taskmaps.items():
                        shim_task = ClassifyShimTask.make_shim(
                            classify_task=ctask,
                            dataset_name=channel)
                        self.register_dataset(shim_task.output())
                        tasks[zi, yi, xi] = shim_task
     
    def generate_block_tasks(self):
        '''Generate tasks that reblock classifications for segmentation'''
        if self.wants_affinity_segmentation:
            old = list(self.classifier_tasks)
            self.classifier_tasks = np.zeros(
                (3, self.n_z, self.n_y, self.n_x), object)
            new = list(self.classifier_tasks)
        else:
            old = [self.classifier_tasks]
            self.classifier_tasks = np.zeros(
                (self.n_z, self.n_y, self.n_x), object)
            new = [self.classifier_tasks]
        if not self.wants_neuroproof_learn:
            if self.wants_transmitter_receptor_synapse_maps:
                old.append(self.transmitter_classifier_tasks)
                self.transmitter_classifier_tasks = \
                    np.zeros((self.n_z, self.n_y, self.n_x), object)
                new.append(self.transmitter_classifier_tasks)
                old.append(self.receptor_classifier_tasks)
                self.receptor_classifier_tasks = \
                    np.zeros((self.n_z, self.n_y, self.n_x), object)
                new.append(self.receptor_classifier_tasks)
            else:
                old.append(self.synapse_classifier_tasks)
                self.synapse_classifier_tasks = \
                    np.zeros((self.n_z, self.n_y, self.n_x), object)
                new.append(self.synapse_classifier_tasks)
            
        old_additional_classifier_tasks = self.additional_classifier_tasks
        self.additional_classifier_tasks = {}
        for name in self.additional_neuroproof_channels:
            if name == SYNAPSE_DATASET and not self.wants_neuroproof_learn:
                self.additional_classifier_tasks[name] =\
                    self.synapse_classifier_tasks
            elif name == SYNAPSE_TRANSMITTER_DATASET \
                 and not self.wants_neuroproof_learn:
                self.additional_classifier_tasks[name] = \
                    self.transmitter_classifier_tasks
            elif name == SYNAPSE_RECEPTOR_DATASET \
                 and not self.wants_neuroproof_learn:
                self.additional_classifier_tasks[name] = \
                    self.receptor_classifier_tasks
            else:
                old.append(old_additional_classifier_tasks)
                self.additional_classifier_tasks[name] = \
                    np.zeros((self.n_z, self.n_y, self.n_x), object)
                new.append(self.additional_classifier_tasks[name])
        for zi in range(self.n_z):
            zs = self.zs[zi]
            ze = self.ze[zi]
            cl_zidx = [idx for idx in range(self.ncl_z)
                       if self.cl_zs[idx] < ze and self.cl_ze[idx] > zs]
            for yi in range(self.n_y):
                ys = self.ys[yi]
                ye = self.ye[yi]
                cl_yidx = [idx for idx in range(self.ncl_y)
                           if self.cl_ys[idx] < ye and self.cl_ye[idx] > ys]
                for xi in range(self.n_x):
                    xs = self.xs[xi]
                    xe = self.xe[xi]
                    cl_xidx = [idx for idx in range(self.ncl_x)
                               if self.cl_xs[idx] < xe and self.cl_xe[idx] > xs]
                    self.generate_block_task(
                        xi, yi, zi, xs, xe, ys, ye, zs, ze, 
                        cl_xidx, cl_yidx, cl_zidx, old, new)
    
    def generate_block_task(self, xi, yi, zi, xs, xe, ys, ye, zs, ze, 
                            cl_xidx, cl_yidx, cl_zidx, old_list, new_list):
        '''Generate the tasks for one x/y/z
        
        xi, yi, zi: indices of the block
        xs, xe, ys, ye, zs, ze: starts and ends of the block
        cl_xidx, cl_yidx, cl_zidx: indices of the overlapping input blocks
        old_list: the old task arrays
        new_list: the new task arrays
        '''
        volume = Volume(xs, ys, zs, xe-xs, ye-ys, ze-zs)
        for old, new in zip(old_list, new_list):
            if old.size == 0:
                continue
            name = old[0, 0, 0].output().dataset_location.dataset_name
            location = self.get_dataset_location(volume, name)
            inputs = []
            tasks = []
            for zz in cl_zidx:
                for yy in cl_yidx:
                    for xx in cl_xidx:
                        task = old[zz, yy, xx]
                        tasks.append(task)
                        tgt = task.output()
                        inputs.append(dict(volume=tgt.volume,
                                           location=tgt.dataset_location))
            if len(inputs) == 1 and volume == inputs[0]["volume"]:
                # A direct block copy - this generally happens on
                # xi = yi = zi = 0.
                # Just carry the old task forward.
                #
                task = old[cl_zidx[0], cl_yidx[0], cl_xidx[0]]
            else:
                task = self.factory.gen_block_task(volume, location, inputs)
                for itask in tasks:
                    task.set_requirement(itask)
            new[zi, yi, xi] = task

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
                        location)
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
                    prob_target = ctask.output()
                    volume = prob_target.volume
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
                    self.register_dataset(stask.output())
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
                        if self.use_distance_watershed:
                            stask.use_distance = True
                            stask.threshold = self.watershed_threshold
                        stask.set_requirement(seeds_task)
                    else:
                        stask = self.factory.gen_2D_segmentation_task(
                            volume=btask.volume,
                            prob_location=prob_location,
                            mask_location=btask.mask_location,
                            seg_location=seg_location,
                            threshold=self.threshold)
                    self.register_dataset(stask.output())
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
                    self.register_dataset(rtask.output())
                    self.resegmentation_tasks[zi, yi, xi] = rtask
    
    def generate_z_watershed_tasks(self):
        '''Generate tasks to go from affinity maps to oversegmentations'''
        self.watershed_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x), object)
        self.segmentation_tasks = self.watershed_tasks
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    ztask = self.classifier_tasks[0, zi, yi, xi]
                    ytask = self.classifier_tasks[1, zi, yi, xi]
                    xtask = self.classifier_tasks[2, zi, yi, xi]
                    volume = ztask.output().volume
                    output_location = self.get_dataset_location(
                        volume, SEG_DATASET)
                    zwtask = self.factory.gen_z_watershed_task(
                         volume=volume,
                         x_prob_location=xtask.output().dataset_location,
                         y_prob_location=ytask.output().dataset_location,
                         z_prob_location=ztask.output().dataset_location,
                         output_location=output_location)
                    zwtask.set_requirement(xtask)
                    zwtask.set_requirement(ytask)
                    zwtask.set_requirement(ztask)
                    zwtask.threshold = self.z_watershed_threshold
                    self.register_dataset(zwtask.output())
                    self.watershed_tasks[zi, yi, xi] = zwtask
    
    def generate_neuroproof_tasks(self):
        '''Generate all tasks involved in Neuroproofing a segmentation
        
        We Neuroproof the blocks and the x, y and z borders between blocks
        '''
        self.np_tasks = np.zeros((self.n_z, self.n_y, self.n_x), object)
        #
        # The task sets are composed of
        # classifier tasks
        # additional classifier tasks
        # segmentation tasks
        # output tasks
        # output dataset name
        #
        if not self.wants_affinity_segmentation:
            task_sets = (
                (self.classifier_tasks,
                 self.additional_classifier_tasks,
                 self.segmentation_tasks,
                 self.np_tasks,
                 NP_DATASET),
            )
        else:
            #
            # Use the x affinity, inverted, as the "membrane"
            # Add the y and z affinities into Neuroproof.
            #
            additional_classifier_tasks = {
                Y_AFFINITY_DATASET:self.classifier_tasks[1],
                Z_AFFINITY_DATASET:self.classifier_tasks[0]
            }
            additional_classifier_tasks.update(self.additional_classifier_tasks)
            task_sets = (
                (self.classifier_tasks[2],
                 additional_classifier_tasks,
                 self.segmentation_tasks,
                 self.np_tasks,
                 NP_DATASET),
            )
            
        for classifier_tasks, additional_classifier_tasks, seg_tasks, np_tasks, \
            dataset_name in task_sets:

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
                        np_task.threshold=self.np_threshold
                        additional_tasks = [ 
                            additional_classifier_tasks[k][zi, yi, xi]
                            for k in self.additional_neuroproof_channels]
                        additional_locations = [
                            task.output().dataset_location for task in
                            additional_tasks]
                        np_task.additional_locations = additional_locations
                        np_task.cpu_count = self.np_cores
                        np_task.set_requirement(classifier_task)
                        np_task.set_requirement(seg_task)
                        np_task.wants_standard_neuroproof =\
                            self.wants_standard_neuroproof
                        map(np_task.set_requirement, additional_tasks)
                        np_tasks[zi, yi, xi] = np_task
                        self.register_dataset(np_task.output())
    
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
            z1 = self.ze[zi]
            for yi in range(self.n_y):
                y0 = self.ys[yi]
                y1 = self.ye[yi]
                for xi in range(self.n_x):
                    x0 = self.xs[xi]
                    x1 = self.xe[xi]
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
                dataset_location,
                resolution=self.resolution)
            self.register_dataset(btask.output())
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
                    location=synapse_gt_location,
                    resolution=self.resolution)
            self.register_dataset(self.gt_synapse_tasks[zi, yi, xi].output())
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
                    location=gt_mask_location,
                    resolution=self.resolution)
            self.register_dataset(self.gt_mask_tasks[zi, yi, xi].output())
            
    def generate_pred_cutouts(self):
        '''Generate volumes matching the ground truth segmentations'''
        self.pred_block_tasks = np.zeros((self.n_z, self.n_y, self.n_x),
                                         object)
        self.gt_block_tasks = np.zeros((self.n_z, self.n_y, self.n_x),
                                       object)
        for zi in range(self.n_z):
            z0 = self.zs[zi] + self.np_z_pad
            z1 = self.ze[zi] - self.np_z_pad
            for yi in range(self.n_y):
                y0 = self.ys[yi] + self.np_y_pad
                y1 = self.ye[yi] - self.np_y_pad
                for xi in range(self.n_x):
                    x0 = self.xs[xi] + self.np_x_pad
                    x1 = self.xe[xi] - self.np_x_pad
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
            json_paths = []
            acc_file = self.all_connected_components_task.output().path
            for zi in range(self.n_z):
                z0 = self.z_grid[zi]
                z1 = self.z_grid[zi+1]
                for yi in range(self.n_y):
                    y0 = self.y_grid[yi]
                    y1 = self.y_grid[yi+1]
                    for xi in range(self.n_x):
                        x0 = self.x_grid[xi]
                        x1 = self.x_grid[xi+1]
                        volume = Volume(x0, y0, z0, x1-x0, y1-y0, z1-z0)
                        ptask = self.np_tasks[zi, yi, xi]
                        gttask = self.gt_tasks[zi, yi, xi]
                        output_location = os.path.join(
                            self.get_dirs(
                                self.xs[xi], self.ys[yi], self.zs[zi])[0],
                            "segmentation_statistics.json")
                        stask = self.factory.gen_segmentation_statistics_task(
                            volume=volume, 
                            gt_seg_location=gttask.output().dataset_location,
                            gt_seg_volume=gttask.output().volume,
                            pred_seg_location=ptask.output().dataset_location,
                            pred_seg_volume=ptask.output().volume,
                            connectivity=acc_file,
                            output_location=output_location)
                        stask.set_requirement(ptask)
                        stask.set_requirement(gttask)
                        stask.set_requirement(
                            self.all_connected_components_task)
                        self.statistics_tasks[zi, yi, xi] = stask
                        json_paths.append(stask.output().path)
            self.statistics_csv_task = self.factory.gen_json_to_csv_task(
                json_paths=json_paths,
                output_path = self.statistics_csv_path,
                excluded_keys=["per_object", "pairs"])
            pdf_path = os.path.splitext(self.statistics_csv_path)[0] + ".pdf"
            self.statistics_report_task = \
                self.factory.gen_segmentation_report_task(
                    json_paths,
                    pdf_path)
            for stask in self.statistics_tasks.flatten():
                self.statistics_report_task.set_requirement(stask)
                self.statistics_csv_task.set_requirement(stask)
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
                            skeleton_location=skel_location,
                            xy_nm=self.xy_nm,
                            z_nm=self.z_nm,
                            decimation_factor=self.skeleton_decimation_factor)
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
            task.joining_method = self.joining_method
            task.min_overlap_percent = self.min_percent_connected
            task.min_overlap_volume = \
                self.min_overlap_volume
            task.max_poly_matches = self.max_poly_matches
            task.dont_join_orphans = self.dont_join_orphans
            task.orphan_min_overlap_ratio = self.orphan_min_overlap_ratio
            task.orphan_min_overlap_volume = self.orphan_min_overlap_volume
        if len(input_tasks) == 0:
            # There's only a single block, so fake doing AllConnectedComponents
            input_task = self.np_tasks[0, 0, 0]
            np_tgt = input_task.output()
            self.all_connected_components_task = \
                FakeAllConnectedComponentsTask(
                    volume=np_tgt.volume,
                    location=np_tgt.dataset_location,
                    output_location=self.get_connectivity_graph_location())
            self.all_connected_components_task.set_requirement(input_task)
            return
        #
        # Build the all-connected-components task
        #
        input_locations = [task.output().path for task in input_tasks]
        self.all_connected_components_task = \
            self.factory.gen_all_connected_components_task(
                input_locations, self.get_connectivity_graph_location())
        for task in input_tasks:
            self.all_connected_components_task.set_requirement(task)
    
    def generate_x_connectivity_graph_tasks(self):
        '''Generate connected components tasks to link blocks in x direction
        
        '''
        self.x_connectivity_graph_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x-1), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x-1):
                    left_task = self.np_tasks[zi, yi, xi]
                    left_tgt = left_task.output()
                    right_task = self.np_tasks[zi, yi, xi+1]
                    right_tgt = right_task.output()
                    #
                    # The overlap is at the average of the x end of the
                    # left block and the x start of the right block
                    #
                    x = self.x_grid[xi+1]
                    overlap_volume = Volume(
                        x-self.halo_size_xy,
                        left_tgt.volume.y,
                        left_tgt.volume.z,
                        self.halo_size_xy * 2 + 1, 
                        left_tgt.volume.height,
                        left_tgt.volume.depth)
                    filename = CONNECTED_COMPONENTS_PATTERN.format(
                        direction="x")
                    output_location = os.path.join(
                            left_tgt.dataset_location.roots[0], filename)
                    task = self.factory.gen_connected_components_task(
                        volume1=left_tgt.volume,
                        location1=left_tgt.dataset_location,
                        volume2=right_tgt.volume,
                        location2=right_tgt.dataset_location,
                        overlap_volume=overlap_volume,
                        output_location=output_location)
                    task.set_requirement(left_task)
                    task.set_requirement(right_task)
                    self.x_connectivity_graph_tasks[zi, yi, xi] = task
                                        
    def generate_y_connectivity_graph_tasks(self):
        '''Generate connected components tasks to link blocks in y direction
        
        '''
        self.y_connectivity_graph_tasks = np.zeros(
            (self.n_z, self.n_y-1, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y - 1):
                for xi in range(self.n_x):
                    left_task = self.np_tasks[zi, yi, xi]
                    left_tgt = left_task.output()
                    right_task = self.np_tasks[zi, yi+1, xi]
                    right_tgt = right_task.output()
                    y = self.y_grid[yi+1]
                    overlap_volume = Volume(
                        left_tgt.volume.x,
                        y - self.halo_size_xy,
                        left_tgt.volume.z,
                        left_tgt.volume.width, 
                        self.halo_size_xy * 2 + 1, 
                        left_tgt.volume.depth)
                    filename = CONNECTED_COMPONENTS_PATTERN.format(
                            direction="y")
                    output_location = os.path.join(
                        left_tgt.dataset_location.roots[0], filename)
                    task = self.factory.gen_connected_components_task(
                        volume1=left_tgt.volume,
                        location1=left_tgt.dataset_location,
                        volume2=right_tgt.volume,
                        location2=right_tgt.dataset_location,
                        overlap_volume=overlap_volume,
                        output_location=output_location)
                    task.set_requirement(left_task)
                    task.set_requirement(right_task)
                    self.y_connectivity_graph_tasks[zi, yi, xi] = task
                                        
    def generate_z_connectivity_graph_tasks(self):
        '''Generate connected components tasks to link blocks in z direction
        
        '''
        self.z_connectivity_graph_tasks = np.zeros(
            (self.n_z-1, self.n_y, self.n_x), object)
        for zi in range(self.n_z-1):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    left_task = self.np_tasks[zi, yi, xi]
                    left_tgt = left_task.output()
                    right_task = self.np_tasks[zi+1, yi, xi]
                    right_tgt = right_task.output()
                    z = self.z_grid[zi+1]
                    overlap_volume = Volume(
                        left_tgt.volume.x,
                        left_tgt.volume.y,
                        z - self.halo_size_z,
                        left_tgt.volume.width, 
                        left_tgt.volume.height, 
                        self.halo_size_z * 2 + 1)
                    filename = CONNECTED_COMPONENTS_PATTERN.format(
                            direction="z")
                    output_location = os.path.join(
                        left_tgt.dataset_location.roots[0], filename)
                    task = self.factory.gen_connected_components_task(
                        volume1=left_tgt.volume,
                        location1=left_tgt.dataset_location,
                        volume2=right_tgt.volume,
                        location2=right_tgt.dataset_location,
                        overlap_volume=overlap_volume,
                        output_location=output_location)
                    task.set_requirement(left_task)
                    task.set_requirement(right_task)
                    self.z_connectivity_graph_tasks[zi, yi, xi] = task    
                    
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
                    self.register_dataset(stask.output())
                    self.synapse_segmentation_tasks[zi, yi, xi] = stask
    
    def generate_synapse_tr_segmentation_tasks(self):
        '''Generate tasks for segmenting synapses with transmitter/receptor maps
        
        '''
        self.synapse_segmentation_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    ttask = self.transmitter_classifier_tasks[zi, yi, xi]
                    rtask = self.receptor_classifier_tasks[zi, yi, xi]
                    nptask = self.np_tasks[zi, yi, xi]
                    volume = ttask.output().volume
                    ttask_loc = ttask.output().dataset_location
                    rtask_loc = rtask.output().dataset_location
                    np_loc = nptask.output().dataset_location
                    stask_loc = self.get_dataset_location(
                        volume, SYN_SEG_DATASET)
                    stask = self.factory.gen_find_synapses_tr_task(
                        volume=volume,
                        transmitter_location=ttask_loc,
                        receptor_location=rtask_loc,
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
                    stask.set_requirement(ttask)
                    stask.set_requirement(rtask)
                    stask.set_requirement(nptask)
                    self.register_dataset(stask.output())
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
                    if self.wants_transmitter_receptor_synapse_maps:
                        #
                        # If we have synapse polarity, hook the polarity
                        # probability maps into the ConnectSynapsesTask
                        #
                        task = self.transmitter_classifier_tasks[zi, yi, xi]
                        sctask.transmitter_probability_map_location = \
                            task.output().dataset_location
                        task = self.receptor_classifier_tasks[zi, yi, xi]
                        sctask.set_requirement(task)
                        sctask.receptor_probability_map_location = \
                            task.output().dataset_location
                        sctask.set_requirement(task)
                    self.synapse_connectivity_tasks[zi, yi, xi] = sctask
        if self.synapse_connection_location != "/dev/null":
            #
            # Generate the task that combines all of the synapse connection
            # files.
            #
            sc_tasks = self.synapse_connectivity_tasks.flatten().tolist()
            sc_outputs = [_.output().path for _ in sc_tasks]
            connectivity_graph_location = self.get_connectivity_graph_location()
            self.aggregate_synapse_connections_task = \
                self.factory.gen_aggregate_connect_synapses_task(
                    sc_outputs, connectivity_graph_location, 
                self.synapse_connection_location)
            map(self.aggregate_synapse_connections_task.set_requirement,
                sc_tasks)
            self.aggregate_synapse_connections_task.set_requirement(
                self.all_connected_components_task)
            yield self.aggregate_synapse_connections_task
            
    
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
        d_gt_neuron_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        d_gt_synapse_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        gt_neuron_synapse_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            z0 = self.zs[zi]
            z1 = self.ze[zi]
            for yi in range(self.n_y):
                y0 = self.ys[yi]
                y1 = self.ye[yi]
                for xi in range(self.n_x):
                    x0 = self.xs[xi]
                    x1 = self.xe[xi]
                    volume=Volume(x0, y0, z0, x1-x0, y1-y0, z1-z0)
                    synapse_gt_task = self.gt_synapse_tasks[zi, yi, xi]
                    synapse_gt_location = \
                        synapse_gt_task.output().dataset_location
                    gt_neuron_task = self.gt_tasks[zi, yi, xi]
                    #
                    # Segment
                    #
                    synapse_gt_seg_location = self.get_dataset_location(
                        volume, SYN_SEG_GT_DATASET)
                    synapse_gt_seg_task = self.factory.gen_cc_segmentation_task(
                        volume=volume,
                        prob_location=synapse_gt_location,
                        mask_location=EMPTY_DATASET_LOCATION,
                        seg_location = synapse_gt_seg_location,
                        threshold=0,
                        dimensionality=Dimensionality.D3,
                        fg_is_higher=True)
                    synapse_gt_seg_task.classes = self.synapse_gt_classes
                    synapse_gt_seg_task.set_requirement(synapse_gt_task)
                    self.register_dataset(synapse_gt_seg_task.output())
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
                    synapse_match_task.min_overlap_pct = \
                        self.synapse_min_overlap_pct
                    synapse_match_task.max_distance = self.synapse_max_distance
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
                    neuron_seg_task = self.np_tasks[zi, yi, xi]
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
                    # Match GT synapses against GT neurons
                    #
                    gt_sn_location = os.path.join(
                        self.get_dirs(x0, y0, z0)[0], "gt_synapse_neuron.json")
                    gt_sn_task = self.factory.gen_connect_synapses_task(
                        volume=volume,
                        neuron_location=gt_neuron_task.output()\
                                                      .dataset_location,
                        synapse_location=synapse_gt_seg_task.output()\
                                                            .dataset_location,
                        xy_dilation=self.gt_neuron_synapse_xy_dilation,
                        z_dilation=self.gt_neuron_synapse_z_dilation,
                        min_contact=self.gt_neuron_synapse_min_contact,
                        output_location=gt_sn_location)
                    gt_sn_task.set_requirement(gt_neuron_task)
                    gt_sn_task.set_requirement(synapse_gt_seg_task)
                    gt_neuron_synapse_tasks[zi, yi, xi] = gt_sn_task
        
        #
        # Create the statistics task
        #
        def locs_of(tasks):
            return [task.output().path for task in tasks]
        statistics_task_location = "synapse-statistics.json"
        synapse_match_tasks = d_gt_synapse_tasks.flatten()
        detected_synapse_connection_tasks = \
            self.synapse_connectivity_tasks.flatten()
        gt_neuron_map_tasks = d_gt_neuron_tasks.flatten()
        #
        # Create the synapse statistics task
        #
        self.synapse_statistics_task = self.factory.gen_synapse_statistics_task(
            locs_of(synapse_match_tasks),
            locs_of(self.synapse_connectivity_tasks.flatten()),
            neuron_map=self.all_connected_components_task.output().path,
            gt_neuron_maps=locs_of(gt_neuron_map_tasks),
            gt_synapse_connections=locs_of(gt_neuron_synapse_tasks.flatten()),
            output_location=self.synapse_statistics_path)
        #
        # Attach the task's dependent tasks
        #
        self.synapse_statistics_task.set_requirement(
            self.all_connected_components_task)
        self.synapse_statistics_task.set_requirement(synapse_gt_task)
        map(self.synapse_statistics_task.set_requirement,
            gt_neuron_synapse_tasks.flatten())
        map(self.synapse_statistics_task.set_requirement, 
            synapse_match_tasks)
        map(self.synapse_statistics_task.set_requirement,
            self.synapse_connectivity_tasks.flatten())
        map(self.synapse_statistics_task.set_requirement,
            gt_neuron_map_tasks)
    
    def get_connectivity_graph_location(self):
        '''Get the location of the AllConnectedComponentsTask output
        
        '''
        if self.connectivity_graph_location != "/dev/null":
            return self.connectivity_graph_location
        elif self.wants_connectivity:
            return os.path.join(self.temp_dirs[0],
                                ALL_CONNECTED_COMPONENTS_JSON)
        return None
    
    def generate_stitched_segmentation_task(self):
        '''Generate the task that builds the HDF5 file with the segmentation
        
        '''
        input_volumes = []
        for task in self.np_tasks.flatten():
            target = task.output()
            input_volumes.append(
                dict(volume=target.volume,
                     location=target.dataset_location))
        location = DatasetLocation(
            [self.stitched_segmentation_location],
            FINAL_SEGMENTATION,
            self.get_pattern(FINAL_SEGMENTATION))
        self.stitched_segmentation_task = \
            self.factory.gen_stitch_segmentation_task(
                input_volumes=input_volumes,
                connected_components_location=
                    self.all_connected_components_task.output().path,
                output_volume=self.volume,
                output_location=location)
        self.stitched_segmentation_task.x_padding = self.np_x_pad
        self.stitched_segmentation_task.y_padding = self.np_y_pad
        self.stitched_segmentation_task.z_padding = self.np_z_pad
        self.stitched_segmentation_task.set_requirement(
            self.all_connected_components_task)
        #
        # These are upstream dependencies of AllConnectedComponentsTask,
        # but add them here anyway to make sure Luigi knows their outputs
        # are needed.
        #
        for task in self.np_tasks.flatten():
            self.stitched_segmentation_task.set_requirement(task)
        self.register_dataset(self.stitched_segmentation_task.output())
    
    ########################################################
    #
    # NPLEARN tasks
    #
    ########################################################
    
    def generate_nplearn_block_tasks(self):
        '''Generate tasks to unify probability masks'''
        if self.wants_affinity_segmentation:
            probability_map_tasks = reversed(list(self.classifier_tasks))
        else:
            probability_map_tasks = [self.classifier_tasks]
        for key in self.additional_neuroproof_channels:
            probability_map_tasks.append(self.additional_classifier_tasks[key])
        self.nplearn_block_tasks = []
        for tasks in probability_map_tasks:
            tasks = tasks.flatten()
            tgt = tasks[0].output()
            name = NPLEARN_PREFIX + tgt.dataset_location.dataset_name
            inputs = [dict(volume=_.output().volume,
                           location=_.output().dataset_location)
                      for _ in tasks]
            output_location = self.get_dataset_location(self.volume, name)
            btask = self.factory.gen_block_task(
                inputs=inputs,
                output_volume=self.volume,
            output_location=output_location)
            map(btask.set_requirement, tasks)
            self.nplearn_block_tasks.append(btask)

    def generate_nplearn_ground_truth_task(self):
        '''Download the ground truth as one big hunk'''
        dataset_location = self.get_dataset_location(
            self.volume, GT_DATASET)
        self.ground_truth_task = self.factory.gen_get_volume_task(
            experiment=self.experiment,
            sample=self.sample,
            dataset=self.dataset,
            channel=self.gt_channel,
            url=self.url,
            volume=self.volume,
            location = dataset_location,
            resolution=self.resolution)
    
    def generate_volume_relabeling_task(self):
        '''Make a global segmentation for nplearn'''
        #
        # I feel so dirty, but faking that the segmentation tasks
        # are the neuroproof tasks means we don't have to rework the
        # connected-components code.
        #
        self.np_tasks = self.segmentation_tasks
        self.generate_connectivity_graph_tasks()
        inputs = [dict(volume=_.output().volume,
                       location=_.output().dataset_location)
                  for _ in self.segmentation_tasks.flatten()]
        output_location = self.get_dataset_location(self.volume,
                                                    NPLEARN_PREFIX+SEG_DATASET)
        self.volume_relabeling_task = \
            self.factory.gen_volume_relabeling_task(
                input_volumes=inputs,
                relabeling_location=
                self.all_connected_components_task.output_location,
                output_volume=self.volume,
                output_location=output_location)
        map(self.volume_relabeling_task.set_requirement,
            self.segmentation_tasks.flatten())
        self.volume_relabeling_task.set_requirement(
            self.all_connected_components_task)
    
    def generate_nplearn_task(self):
        '''Make the task that trains neuroproof'''
        prob_location = self.nplearn_block_tasks[0].output().dataset_location
        seg_location = self.volume_relabeling_task.output().dataset_location
        gt_location = self.ground_truth_task.output().dataset_location
        self.neuroproof_learn_task = \
            self.factory.gen_neuroproof_learn_task(
                volume=self.volume,
                prob_location=prob_location,
                seg_location=seg_location,
                gt_location=gt_location,
                output_location=self.neuroproof_classifier_path,
                strategy=self.nplearn_strategy,
                num_iterations=self.nplearn_num_iterations,
                prune_feature=self.prune_feature,
                use_mito=self.use_mito) 
        self.neuroproof_learn_task.cpu_count = self.nplearn_cpu_count
        map(self.neuroproof_learn_task.set_requirement,
            self.nplearn_block_tasks)
        self.neuroproof_learn_task.set_requirement(self.volume_relabeling_task)
        self.neuroproof_learn_task.set_requirement(self.ground_truth_task)
    
    def register_dataset(self, target):
        '''Register the location of a dataset
        
        :param target: a VolumeTarget having a volume and a dataset_location
        
        Store the location of a target produced by the pipeline in the
        datasets dictionary for inclusion in the index file.
        '''
        dataset_name = target.dataset_location.dataset_name
        if dataset_name not in self.datasets:
            self.datasets[dataset_name] = {}
        volume = to_hashable(target.volume.to_dictionary())
        self.datasets[dataset_name][volume] = \
            target.dataset_location.to_dictionary()
        
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
            self.requirements = []
            self.datasets = {}
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
                if self.wants_neuroproof_learn:
                    rh_logger.logger.report_event(
                        "Making GT Butterfly task for nplearn")
                    self.generate_nplearn_ground_truth_task()
                else:
                    rh_logger.logger.report_event("Making gt cutout tasks")
                    self.generate_gt_cutouts()
                #
                # Step 2: run the pixel classifier on each
                #
                rh_logger.logger.report_event("Making classifier tasks")
                self.generate_classifier_tasks()
                #
                # Step 3: reblock the classification results for overlapped
                #         segmentation
                #
                rh_logger.logger.report_event("Making block tasks")
                self.generate_block_tasks()
                if not self.wants_affinity_segmentation:
                    #
                    # Step 4: make the border masks
                    #
                    rh_logger.logger.report_event("Making border mask tasks")
                    self.generate_border_mask_tasks()
                    if self.method != SeedsMethodEnum.ConnectedComponents:
                        #
                        # Step 5: find the seeds for the watershed
                        #
                        rh_logger.logger.report_event(
                            "Making watershed seed tasks")
                        self.generate_seed_tasks()
                    #
                    # Step 6: run watershed
                    #
                    rh_logger.logger.report_event("Making watershed tasks")
                    self.generate_watershed_tasks()
                    if self.wants_resegmentation:
                        self.generate_resegmentation_tasks()
                else:
                    #
                    # For affinity maps, run the z-watershed to produce
                    # the oversegmentation
                    #
                    rh_logger.logger.report_event("Making z-watershed tasks")
                    self.generate_z_watershed_tasks()
                if self.wants_neuroproof_learn:
                    #
                    # Neuroproof Learn needs to have everything reblocked
                    #
                    rh_logger.logger.report_event(
                        "Making segmentation relabeling task")
                    self.generate_volume_relabeling_task()
                    rh_logger.logger.report_event(
                        "Making tasks to reblock probability maps for nplearn")
                    self.generate_nplearn_block_tasks()
                    rh_logger.logger.report_event(
                        "Making neuroproof learn task")
                    self.generate_nplearn_task()
                    self.requirements.append(self.neuroproof_learn_task)
                    return
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
                if self.wants_transmitter_receptor_synapse_maps:
                    self.generate_synapse_tr_segmentation_tasks()
                else:
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
                requirements = self.generate_synapse_connectivity_tasks()
                self.requirements += list(requirements)
                #
                # Step 12: find ground-truth synapses and compute statistics
                #
                rh_logger.logger.report_event("Comparing synapses to gt")
                self.generate_synapse_statistics_tasks()
                #
                # Step 13: write out the stitched segmentation
                #
                if self.stitched_segmentation_location != "/dev/null":
                    rh_logger.logger.report_event("Stitching segments")
                    self.generate_stitched_segmentation_task()
                    self.requirements.append(self.stitched_segmentation_task)
                #
                # The requirements:
                #
                # The skeletonize tasks if skeletonization is done
                #     otherwise the block neuroproof tasks
                # The border neuroproof tasks
                # The statistics task
                #
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
        
    def ariadne_run(self):
        '''Write the optional index file'''
        if self.index_file_location != "/dev/null":
            for key in self.datasets:
                self.datasets[key] = \
                    [(dict(k), v) for k, v in self.datasets[key].items()]
            self.datasets["experiment"] = self.experiment
            self.datasets["sample"] = self.sample
            self.datasets["dataset"] = self.dataset
            self.datasets["channel"] = self.channel
            json.dump(self.datasets, open(self.index_file_location, "w"))

class PipelineTask(PipelineTaskMixin, PipelineRunReportMixin, luigi.Task):
    task_namespace = "ariadne_microns_pipeline"
    
        
        
        
