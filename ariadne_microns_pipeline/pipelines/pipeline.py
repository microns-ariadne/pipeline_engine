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
from ..targets.volume_target import write_loading_plan, write_storage_plan
from ..targets.butterfly_target import ButterflyChannelTarget
from ..parameters import Volume, VolumeParameter
from ..parameters import EMPTY_LOCATION, EMPTY_DATASET_ID, is_empty_dataset_id
from ..parameters import DEFAULT_LOCATION
from ..tasks.utilities import to_hashable
from ..volumedb import VolumeDB, Persistence, UINT8, UINT16, UINT32
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

'''The name of the synapse gt dataset'''
SYN_GT_DATASET = "synapse-gt"

'''The name of the ground-truth annotation mask dataset'''
GT_MASK_DATASET = "gt-mask"

'''The name of the segmentation of the synapse gt dataset'''
SYN_SEG_GT_DATASET = "synapse-gt-segmentation"

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
    temp_dir = luigi.Parameter(
        description="The root directory for ephemeral and intermediate data")
    root_dir = luigi.Parameter(
        description="The root directory for data to be saved")
    #########
    #
    # Optional parameters
    #
    #########
    datatypes_to_keep = luigi.ListParameter(
        default=[],
        description="Names of the datasets (e.g. \"neuroproof\") to store "
        "under the root directory.")
    volume_db_url = luigi.Parameter(
        default=DEFAULT_LOCATION,
        description="The sqlalchemy URL to use to connect to the volume "
        "database")
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
        default=EMPTY_LOCATION)
    synapse_statistics_path = luigi.Parameter(
        default=EMPTY_LOCATION,
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
        default=EMPTY_LOCATION,
        description="The location of the all-connected-components connectivity"
                    " .json file. Default = do not generate it.")
    
    stitched_segmentation_location = luigi.Parameter(
        default=EMPTY_LOCATION,
        description="The location for the final stitched segmentation")
    index_file_location = luigi.Parameter(
        default=EMPTY_LOCATION,
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
        default=EMPTY_LOCATION,
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

    def get_dir(self, x, y, z):
        '''Return a directory suited for storing a file with the given offset
        
        Create a hierarchy of directories in order to limit the number
        of files in any one directory.
        '''
        return os.path.join(self.temp_dir,
                            self.experiment,
                            self.sample,
                            self.dataset,
                            self.channel,
                            str(x),
                            str(y),
                            str(z))
    
    @property
    def wants_connectivity(self):
        '''True if we are doing a connectivity graph'''
        return self.connectivity_graph_location != EMPTY_LOCATION or \
            self.stitched_segmentation_location != EMPTY_LOCATION or \
            self.wants_neuron_statistics or \
            self.wants_synapse_statistics or \
            self.wants_neuroproof_learn
    
    @property
    def wants_neuron_statistics(self):
        '''True if we want to calculate neuron segmentation accuracy'''
        return self.statistics_csv_path != EMPTY_LOCATION
    
    @property
    def wants_synapse_statistics(self):
        '''True if we are scoring synapses against ground truth'''
        return self.synapse_statistics_path != EMPTY_LOCATION
    
    @property
    def has_annotation_mask(self):
        '''True if there is a mask of the ground-truth annotated volume'''
        return self.gt_mask_channel != NO_CHANNEL

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
        self.register_datatype(SEG_DATASET, UINT32, 
                               "An oversegmentation, before Neuroproof")
        self.register_datatype(
            SYN_SEG_DATASET, UINT16,
            "A segmentation of a synapse")
        self.register_datatype(
            FILTERED_SYN_SEG_DATASET, UINT16,
            "The segmentation of synapses after running a filtering algorithm "
            "to remove false positives")
        self.register_datatype(SEEDS_DATASET, UINT32,
                              "The seeds for a watershed segmentation")
        self.register_datatype(MASK_DATASET, UINT8,
                              "The mask of the extra-cellular space")
        self.register_datatype(IMG_DATASET, UINT8,
                              "The initial raw image data")
        self.register_datatype(MEMBRANE_DATASET, UINT8,
                              "Membrane probabilities")
        self.register_datatype(X_AFFINITY_DATASET, UINT8,
                               "Probabilities of connecting pixels in the "
                               "X direction")
        self.register_datatype(Y_AFFINITY_DATASET, UINT8,
                                   "Probabilities of connecting pixels in the "
                                   "Y direction")
        self.register_datatype(Z_AFFINITY_DATASET, UINT8,
                                   "Probabilities of connecting pixels in the "
                                   "Z direction")
        self.register_datatype(SYNAPSE_DATASET, UINT8,
                               "Probability of a voxel being in a synapse")
        self.register_datatype(SYNAPSE_TRANSMITTER_DATASET, UINT8,
                               "Probability of a voxel being in the "
                               "presynaptic partner of a synapse")
        self.register_datatype(SYNAPSE_RECEPTOR_DATASET, UINT8,
                               "Probability of a voxel being in the "
                               "postsynaptic partner of a synapse")
        self.register_datatype(NP_DATASET, UINT32,
                               "The neuroproofed segmentation")
        self.register_datatype(GT_DATASET, UINT32,
                               "The ground-truth segmentation")
        self.register_datatype(SYN_GT_DATASET, UINT8, 
                               "The markup for synapse ground-truth")
        self.register_datatype(SYN_SEG_GT_DATASET, UINT32,
                               "The ground-truth synapse segmentation")
        
        
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
        '''Get volumes padded for classifier'''
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
                    task = self.factory.gen_get_volume_task(
                            experiment=self.experiment,
                            sample=self.sample,
                            dataset=self.dataset,
                            channel=self.channel,
                            url=self.url,
                            volume=volume,
                            dataset_name=IMG_DATASET,
                            resolution=self.resolution)
                    self.datasets[task.output().path] = task
                    self.tasks.append(task)

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
            datasets = {
                self.x_affinity_class_name:X_AFFINITY_DATASET,
                self.y_affinity_class_name:Y_AFFINITY_DATASET,
                self.z_affinity_class_name:Z_AFFINITY_DATASET
                }
        else:
            datasets = {
                self.membrane_class_name: MEMBRANE_DATASET
            }
            
        if not self.wants_neuroproof_learn:
            if self.wants_transmitter_receptor_synapse_maps:
                datasets.update({
                    self.transmitter_class_name: SYNAPSE_TRANSMITTER_DATASET, 
                    self.receptor_class_name: SYNAPSE_RECEPTOR_DATASET
                })
            else:
                datasets.update({
                    self.synapse_class_name: SYNAPSE_DATASET})
        for channel in self.additional_neuroproof_channels:
            if channel not in (SYNAPSE_DATASET, SYNAPSE_TRANSMITTER_DATASET,
                               SYNAPSE_RECEPTOR_DATASET) \
               and not self.wants_neuroproof_learn:
                datasets[channel] = channel
        for zi in range(self.ncl_z):
            for yi in range(self.ncl_y):
                for xi in range(self.ncl_x):
                    x0 = self.cl_xs[xi]
                    x1 = self.cl_xe[xi]
                    y0 = self.cl_ys[yi]
                    y1 = self.cl_ye[yi]
                    z0 = self.cl_zs[zi]
                    z1 = self.cl_ze[zi]
                    output_volume = Volume(x0, y0, z0, 
                                           x1-x0, y1-y0, z1-z0)
                    x0p = x0 - self.nn_x_pad
                    x1p = x1 + self.nn_x_pad
                    y0p = y0 - self.nn_y_pad
                    y1p = y1 + self.nn_y_pad
                    z0p = z0 - self.nn_z_pad
                    z1p = z1 + self.nn_z_pad
                    input_volume = Volume(x0p, y0p, z0p,
                                          x1p - x0p, y1p - y0p, z1p - z0p)
                    ctask = self.factory.gen_classify_task(
                        datasets=datasets,
                        img_volume=input_volume,
                        output_volume=output_volume,
                        dataset_name=IMG_DATASET,
                        classifier_path=self.pixel_classifier_path)
                    self.tasks.append(ctask)
                    #
                    # Create shims for all channels
                    #
                    for channel in datasets.values():
                        shim_task = ClassifyShimTask.make_shim(
                            classify_task=ctask,
                            dataset_name=channel)
                        self.datasets[shim_task.output().path] = shim_task
                        self.tasks.append(shim_task)
    
    def get_block_volume(self, xi, yi, zi):
        '''Get the volume for a segmentation block
        
        :param xi: the X index of the block
        :param yi: the Y index of the block
        :param zi: the Z index of the block
        '''
        return Volume(self.xs[xi], self.ys[yi], self.zs[zi],
                      self.xe[xi] - self.xs[xi],
                      self.ye[yi] - self.ys[yi],
                      self.ze[zi] - self.zs[zi])
    
    def generate_border_mask_tasks(self):
        '''Create a border mask for each block'''
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    volume = self.get_block_volume(xi, yi, zi)
                    btask = self.factory.gen_mask_border_task(
                        volume,
                        MEMBRANE_DATASET,
                        MASK_DATASET)
                    self.datasets[btask.output().path] = btask
                    self.tasks.append(btask)
                    
    def generate_seed_tasks(self):
        '''Find seeds for the watersheds'''
        self.seed_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    volume = self.get_block_volume(xi, yi, zi)
                    stask = self.factory.gen_find_seeds_task(
                        volume=volume,
                        prob_dataset_name=MEMBRANE_DATASET,
                        seeds_dataset_name=SEEDS_DATASET,
                        sigma_xy=self.sigma_xy, 
                        sigma_z=self.sigma_z, 
                        threshold=self.threshold,
                        method=self.method,
                        dimensionality=self.dimensionality,
                        minimum_distance_xy=self.minimum_distance_xy,
                        minimum_distance_z=self.minimum_distance_z)
                    self.datasets[stask.output().path] = stask
                    self.tasks.append(stask)
                    self.seed_tasks[zi, yi, xi] = stask
                    
    def generate_watershed_tasks(self):
        '''Run watershed on each pixel '''
        self.watershed_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x), object)
        self.segmentation_tasks = self.watershed_tasks
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    volume = self.get_block_volume(xi, yi, zi)
                    seeds_task = self.seed_tasks[zi, yi, xi]
                    stask = self.factory.gen_segmentation_task(
                        volume=volume,
                        prob_dataset_name=MEMBRANE_DATASET,
                        mask_dataset_name=MASK_DATASET,
                        seeds_dataset_name=SEEDS_DATASET,
                        seeds_src_task=seeds_task,
                        seg_dataset_name=SEG_DATASET,
                        sigma_xy=self.sigma_xy,
                        sigma_z=self.sigma_z,
                        dimensionality=self.dimensionality)
                    if self.use_distance_watershed:
                        stask.use_distance = True
                        stask.threshold = self.watershed_threshold
                    self.watershed_tasks[zi, yi, xi] = stask
                    self.datasets[stask.output().path] = stask
                    self.tasks.append(stask)
    
    def generate_resegmentation_tasks(self):
        self.resegmentation_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x), object)
        self.segmentation_tasks = self.resegmentation_tasks
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    volume = self.get_block_volume(xi, yi, zi)
                    wtask = self.watershed_tasks[zi, yi, xi]
                    rtask = self.factory.gen_unsegmentation_task(
                        volume = volume,
                        input_dataset_name=SEG_DATASET,
                        output_dataset_name=RESEG_DATASET,
                        use_min_contact = self.use_min_contact,
                        contact_threshold = self.contact_threshold)
                    self.resegmentation_tasks[zi, yi, xi] = rtask
                    self.datasets[rtask.output().plan] = rtask
                    self.tasks.append(rtask)
    
    def generate_z_watershed_tasks(self):
        '''Generate tasks to go from affinity maps to oversegmentations'''
        self.watershed_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x), object)
        self.segmentation_tasks = self.watershed_tasks
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    volume = self.get_block_volume(xi, yi, zi)
                    zwtask = self.factory.gen_z_watershed_task(
                         volume=volume,
                         x_prob_dataset_name=X_AFFINITY_DATASET,
                         y_prob_dataset_name=Y_AFFINITY_DATASET,
                         z_prob_dataset_name=Z_AFFINITY_DATASET,
                         output_dataset_name=SEG_DATASET)
                    zwtask.threshold = self.z_watershed_threshold
                    self.watershed_tasks[zi, yi, xi] = zwtask
                    self.datasets[zwtask.output().path] = zwtask
                    self.tasks.append(zwtask)
    
    def generate_neuroproof_tasks(self):
        '''Generate neuroproof tasks for all blocks
        
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
            prob_dataset_name = MEMBRANE_DATASET
            additional_dataset_names = []
        else:
            prob_dataset_name = X_AFFINITY_DATASET
            additional_dataset_names = [
                Y_AFFINITY_DATASET, Z_AFFINITY_DATASET]
        additional_dataset_names += self.additional_neuroproof_channels
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    volume = self.get_block_volume(xi, yi, zi)
                    src_task = self.segmentation_tasks[zi, yi, xi]
                    np_task = self.factory.gen_neuroproof_task(
                        volume=volume,
                        prob_dataset_name=prob_dataset_name,
                        additional_dataset_names=additional_dataset_names,
                        input_seg_dataset_name=SEG_DATASET,
                        output_dataset_name=NP_DATASET,
                        classifier_filename=self.neuroproof_classifier_path,
                        input_seg_src_task=src_task)
                    np_task.threshold=self.np_threshold
                    self.np_tasks[zi, yi, xi] = np_task
                    self.datasets[np_task.output().path] = np_task
                    self.tasks.append(np_task)
    
    def generate_gt_cutouts(self):
        '''Generate volumes of ground truth segmentation
        
        Get the ground-truth neuron data, the synapse data if needed and
        the mask of the annotated area, if present
        '''
        if self.wants_neuron_statistics or self.wants_synapse_statistics or \
           self.wants_neuroproof_learn:
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
        if self.wants_neuron_statistics or self.wants_synapse_statistics or \
           self.wants_neuroproof_learn:
            btask = self.factory.gen_get_volume_task(
                self.experiment,
                self.sample,
                self.dataset,
                self.gt_channel,
                self.url,
                volume,
                GT_DATASET,
                resolution=self.resolution)
            self.datasets[btask.output().path] = btask
        if self.wants_synapse_statistics:
            stask =\
                self.factory.gen_get_volume_task(
                    experiment=self.experiment,
                    sample=self.sample,
                    dataset=self.dataset,
                    channel=self.synapse_channel,
                    url=self.url,
                    volume=volume,
                    dataset_name=SYN_GT_DATASET,
                    resolution=self.resolution)
            self.datasets[stask.output().path] = stask
        if self.has_annotation_mask:
            mtask = self.factory.gen_get_volume_task(
                    experiment=self.experiment,
                    sample=self.sample,
                    dataset=self.dataset,
                    channel=self.gt_mask_channel,
                    url=self.url,
                    volume=volume,
                    dataset_name=GT_MASK_DATASET,
                    resolution=self.resolution)
            self.datasets[mtask.output().path] = mtask
            self.tasks.append(mtask)
            
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
                        output_location = os.path.join(
                            self.get_dir(
                                self.xs[xi], self.ys[yi], self.zs[zi]),
                            "segmentation_statistics.json")
                        stask = self.factory.gen_segmentation_statistics_task(
                            volume=volume, 
                            gt_seg_dataset_name=GT_DATASET,
                            pred_seg_dataset_name=NP_DATASET,
                            connectivity=acc_file,
                            output_location=output_location,
                            pred_src_task=ptask)
                        self.tasks.append(stask)
                        stask.set_requirement(
                            self.all_connected_components_task)
                        self.datasets[stask.output().path] = stask
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
                        skel_root = self.get_dir(self.xs[xi],
                                                 self.ys[yi],
                                                 self.zs[zi])
                        skel_location = os.path.join(
                            skel_root, SKEL_DIR_NAME)
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
                            os.dirname(left_tgt.output().path), filename)
                    task = self.factory.gen_connected_components_task(
                        dataset_name=NP_DATASET,
                        volume1=left_tgt.volume,
                        src_task1=left_task,
                        volume2=right_tgt.volume,
                        src_task_2=right_task,
                        overlap_volume=overlap_volume,
                        output_location=output_location)
                    self.tasks.append(task)
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
                        os.path.dirname(left_tgt.output().path), filename)
                    task = self.factory.gen_connected_components_task(
                        dataset_name=NP_DATASET,
                        volume1=left_tgt.volume,
                        src_task1=left_task,
                        volume2=right_tgt.volume,
                        src_task_2=right_task,
                        overlap_volume=overlap_volume,
                        output_location=output_location)
                    self.tasks.append(task)
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
                        os.path.dirname(left_tgt.output().path), filename)
                    task = self.factory.gen_connected_components_task(
                        volume1=left_tgt.volume,
                        src_task1=left_task,
                        volume2=right_tgt.volume,
                        src_task2=right_task,
                        overlap_volume=overlap_volume,
                        output_location=output_location)
                    self.z_connectivity_graph_tasks[zi, yi, xi] = task
                    self.tasks.append(task)
                    
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
                    volume = self.get_block_volume(xi, yi, zi)
                    stask = self.factory.gen_find_synapses_task(
                        volume=volume,
                        synapse_prob_dataset_name=SYNAPSE_DATASET,
                        neuron_segmentation_dataset_name=NP_DATASET,
                        output_dataset_name=SYN_SEG_DATASET,
                        threshold=self.synapse_threshold,
                        erosion_xy=self.synapse_xy_erosion,
                        erosion_z=self.synapse_z_erosion,
                        sigma_xy=self.synapse_xy_sigma,
                        sigma_z=self.synapse_z_sigma,
                        min_size_2d=self.synapse_min_size_2d,
                        max_size_2d=self.synapse_max_size_2d,
                        min_size_3d=self.min_synapse_area,
                        min_slice=self.min_synapse_depth)
                    self.tasks.append(stask)
                    self.datasets[stask.output().path] = stask
                    self.synapse_segmentation_tasks[zi, yi, xi] = stask
    
    def generate_synapse_tr_segmentation_tasks(self):
        '''Generate tasks for segmenting synapses with transmitter/receptor maps
        
        '''
        self.synapse_segmentation_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    volume = self.get_block_volume(xi, yi, zi)
                    stask = self.factory.gen_find_synapses_tr_task(
                        volume=volume,
                        transmitter_dataset_name=SYNAPSE_TRANSMITTER_DATASET,
                        receptor_dataset_name=SYNAPSE_RECEPTOR_DATASET,
                        neuron_dataset_name=NP_DATASET,
                        output_dataset_name=SYN_SEG_DATASET,
                        threshold=self.synapse_threshold,
                        erosion_xy=self.synapse_xy_erosion,
                        erosion_z=self.synapse_z_erosion,
                        sigma_xy=self.synapse_xy_sigma,
                        sigma_z=self.synapse_z_sigma,
                        min_size_2d=self.synapse_min_size_2d,
                        max_size_2d=self.synapse_max_size_2d,
                        min_size_3d=self.min_synapse_area,
                        min_slice=self.min_synapse_depth)
                    self.tasks.append(stask)
                    self.datasets[stask.output().path] = stask
                    self.synapse_segmentation_tasks[zi, yi, xi] = stask
                    
    
    def generate_synapse_connectivity_tasks(self):
        '''Make tasks that connect neurons to synapses'''
        #
        # TODO: use the global segmentation
        #
        self.synapse_connectivity_tasks = np.zeros(
            (self.n_z, self.n_y, self.n_x), object)
        if self.wants_transmitter_receptor_synapse_maps:
            transmitter_dataset_name = SYNAPSE_TRANSMITTER_DATASET
            receptor_dataset_name = SYNAPSE_RECEPTOR_DATASET
        else:
            transmitter_dataset_name = receptor_dataset_name = None
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    volume = self.get_block_volume(xi, yi, zi)
                    segtask = self.synapse_segmentation_tasks[zi, yi, xi]
                    ntask = self.np_tasks[zi, yi, xi]
                    output_location = os.path.join(
                        self.get_dir(
                            self.xs[xi], self.ys[yi], self.zs[zi]),
                        "synapse_connectivity.json")
                    
                    sctask = self.factory.gen_connect_synapses_task(
                        volume=volume,
                        synapse_dataset_name=SYN_SEG_DATASET,
                        neuron_dataset_name=NP_DATASET,
                        transmitter_dataset_name=transmitter_dataset_name,
                        receptor_dataset_name=receptor_dataset_name,
                        output_location=output_location,
                        xy_dilation=self.synapse_xy_dilation,
                        z_dilation=self.synapse_z_dilation,
                        min_contact=self.min_synapse_neuron_contact,
                        synapse_src_task=segtask,
                        neuron_src_task=ntask)
                    self.tasks.append(sctask)
                    self.synapse_connectivity_tasks[zi, yi, xi] = sctask
        if self.synapse_connection_location != EMPTY_LOCATION:
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
        if self.has_annotation_mask:
            mask_dataset_name = GT_MASK_DATASET
        else:
            mask_dataset_name = None
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    volume = self.get_block_volume(xi, yi, zi)
                    #
                    # Segment
                    #
                    synapse_gt_seg_task = self.factory.gen_cc_segmentation_task(
                        volume=volume,
                        prob_dataset_name=SYN_GT_DATASET,
                        seg_dataset_name = SYN_SEG_GT_DATASET,
                        threshold=0,
                        dimensionality=Dimensionality.D3,
                        fg_is_higher=True)
                    synapse_gt_seg_task.classes = self.synapse_gt_classes
                    self.datasets[synapse_gt_seg_task.output().path] = \
                        synapse_gt_seg_task
                    self.tasks.append(synapse_gt_seg_task)
                    #
                    # Match GT synapses against detected synapses
                    #
                    synapse_seg_task = \
                        self.synapse_segmentation_tasks[zi, yi, xi]
                    synapse_match_location = os.path.join(
                        self.get_dir(x0, y0, z0), "synapse-match.json")
                    synapse_match_task = self.factory.gen_match_synapses_task(
                        volume=volume,
                        gt_dataset_name=SYN_SEG_GT_DATASET,
                        detected_dataset_name=SYN_SEG_DATASET,
                        detected_src_task=synapse_seg_task,
                        mask_dataset_name=mask_dataset_name,
                        output_location=synapse_match_location,
                        method=self.synapse_match_method)
                    synapse_match_task.min_overlap_pct = \
                        self.synapse_min_overlap_pct
                    synapse_match_task.max_distance = self.synapse_max_distance
                    synapse_match_task.set_requirement(synapse_seg_task)
                    synapse_match_task.set_requirement(synapse_gt_seg_task)
                    d_gt_synapse_tasks[zi, yi, xi] = synapse_match_task
                    self.tasks.append(synapse_match_task)
                    #
                    # Match GT neurons against detected neurons
                    #
                    neuron_match_location = os.path.join(
                        self.get_dir(x0, y0, z0), "neuron-match.json")
                    neuron_seg_task = self.np_tasks[zi, yi, xi]
                    neuron_match_task=self.factory.gen_match_neurons_task(
                        volume=volume,
                        gt_dataset_name=GT_DATASET,
                        detected_dataset_name=NP_DATASET,
                        detected_src_task=neuron_seg_task,
                        output_location=neuron_match_location)
                    d_gt_neuron_tasks[zi, yi, xi] = neuron_match_task
                    self.tasks.append(neuron_match_task)
                    #
                    # Match GT synapses against GT neurons
                    #
                    gt_sn_location = os.path.join(
                        self.get_dir(x0, y0, z0), "gt_synapse_neuron.json")
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
        if self.connectivity_graph_location != EMPTY_LOCATION:
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
        self.tasks.append(self.stitched_segmentation_task)
    
    ########################################################
    #
    # NPLEARN tasks
    #
    ########################################################
    
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
    
    def compute_requirements(self):
        '''Compute the requirements for this task'''
        if not hasattr(self, "requirements"):
            import cProfile
            import pstats
            pr = cProfile.Profile()
            pr.enable()
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
            self.tasks = []
            try:
                if not os.path.isdir(self.root_dir):
                    os.makedirs(self.root_dir)
                self.init_db()
                self.factory = AMTaskFactory(self.volume_db_url, 
                                             self.volume_db)
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
                    # Neuroproof Learn needs to have the segmentation relabeled
                    #
                    rh_logger.logger.report_event(
                        "Making segmentation relabeling task")
                    self.generate_volume_relabeling_task()
                    rh_logger.logger.report_event(
                        "Making neuroproof learn task")
                    self.generate_nplearn_task()
                    self.requirements.append(self.neuroproof_learn_task)
                else:
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
                    if self.stitched_segmentation_location != EMPTY_LOCATION:
                        rh_logger.logger.report_event("Stitching segments")
                        self.generate_stitched_segmentation_task()
                        self.requirements.append(self.stitched_segmentation_task)
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
                for task in self.tasks:
                    for tgt in task.input():
                        path = tgt.path
                        if path in self.datasets:
                            task.set_requirement(self.datasets[path])
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
            finally:
                with open("/tmp/profile.log", "w") as fd:
                    pr.disable()
                    ps = pstats.Stats(pr, stream=fd).sort_stats("cumulative")
                    ps.print_stats()
    
    def requires(self):
        self.compute_requirements()
        return self.requirements
        
    def ariadne_run(self):
        '''Write the optional index file'''
        if self.index_file_location != EMPTY_LOCATION:
            for key in self.datasets:
                self.datasets[key] = \
                    [(dict(k), v) for k, v in self.datasets[key].items()]
            json.dump(self.datasets, open(self.index_file_location, "w"))

class PipelineTask(PipelineTaskMixin, PipelineRunReportMixin, luigi.Task):
    task_namespace = "ariadne_microns_pipeline"
    
        
        
        
