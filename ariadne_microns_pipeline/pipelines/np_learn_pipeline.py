'''The NeuroproofLearnPipeline trains Neuroproof on a segmentation

The pipeline reads a volume from Butterfly, classifies it, segments it and
then runs Neuroproof on the classifier prediction and segmented volume.
'''
import luigi
import numpy as np
import os
import rh_logger
import tempfile

from .utilities import PipelineRunReportMixin
from ..targets.butterfly_target import ButterflyChannelTarget
from ..tasks import AMTaskFactory
from ..tasks.classify import ClassifyShimTask
from ..tasks.find_seeds import Dimensionality, SeedsMethodEnum
from ..tasks.nplearn import StrategyEnum
from ..targets.classifier_target import PixelClassifierTarget
from ..parameters import DatasetLocation, Volume, VolumeParameter

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

'''The name of the ground-truth dataset for statistics computation'''
GT_DATASET = "gt"

'''The name of the global membrane prediction'''
GLOBAL_MEMBRANE_DATASET = "global_membrane"

'''The name of the global segmentation'''
GLOBAL_SEGMENTATION_DATASET = "global_segmentation"

class NeuroproofLearnPipelineTaskMixin:
    
    #########
    #
    # Butterfly parameters
    #
    #########
    
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

    #########
    #
    # Pixel classifier
    #
    #########
    
    pixel_classifier_path = luigi.Parameter(
        description="Path to pickled pixel classifier")
    
    #########
    #
    # Target volume
    #
    #########
    
    volume = VolumeParameter(
        description="The volume to segment")

    #########
    #
    # The neuroproof classifier
    #
    #########
    output_location = luigi.Parameter(
        description="Location for the classifier file. Use an .xml extension "
        "to use the OpenCV random forest classifier. Use an .h5 extension "
        "to use the Vigra random forest classifier")
    neuroproof = luigi.Parameter(
        description="Location of the neuroproof_graph_learn binary")
    neuroproof_ld_library_path = luigi.Parameter(
        description="Library paths to Neuroproof's shared libraries. "
        "This should include paths to CILK stdc++ libraries, Vigra libraries, "
        "JSONCPP libraries, and OpenCV libraries.")
    strategy = luigi.EnumParameter(
        enum=StrategyEnum,
        default=StrategyEnum.all,
        description="Learning strategy to use")
    num_iterations = luigi.IntParameter(
        default=1,
        description="Number of iterations used for learning")
    prune_feature = luigi.BoolParameter(
        description="Automatically prune useless features")
    use_mito = luigi.BoolParameter(
        description="Set delayed mito agglomeration")
    
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
    block_xy_overlap = luigi.IntParameter(
        description="# of pixels of overlap between adjacent blocks "
        "in the X/Y direction",
        default=20)
    block_z_overlap = luigi.IntParameter(
        description="# of pixels of overlap between adjacent blocks in the "
        "z direction",
        default=5)
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
    temp_dirs = luigi.ListParameter(
        description="The base location for intermediate files",
        default=(tempfile.gettempdir(),))

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
    
    def get_connected_components_location(self, x0, x1, y0, y1, z0, z1):
        '''Get the location of the file that links two segmentations'''
        return os.path.join(self.temp_dirs[0],
                            "%d-%d_%d-%d_%d-%d.json" %
                            (x0, x1, y0, y1, z0, z1))
    
    def get_all_connected_components_location(self):
        '''Get the location for the output of AllConnectedComponentsTask'''
        return os.path.join(self.temp_dirs[0], "all_connected_components.json")
    
    def get_segmentation_location(self):
        '''Get the location of the rewritten, combined segmentation'''
        return DatasetLocation(self.get_dirs(self.volume.x,
                                             self.volume.y,
                                             self.volume.z),
                               GLOBAL_SEGMENTATION_DATASET,
                               self.get_pattern(GLOBAL_SEGMENTATION_DATASET))
    
    def get_global_prediction_location(self):
        '''Get the location for the reblocked unified membrane prediction'''
        return DatasetLocation(self.get_dirs(self.volume.x,
                                             self.volume.y,
                                             self.volume.z),
                               GLOBAL_MEMBRANE_DATASET,
                               self.get_pattern(GLOBAL_MEMBRANE_DATASET))
    
    def compute_requirements(self):
        '''Compute the requirements for this task'''
        if not hasattr(self, "requirements_computed"):
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
                # Step 6: compute connected components between overlapping
                #         blocks
                #
                rh_logger.logger.report_event(
                    "Making connected components tasks")
                self.generate_x_connected_components_tasks()
                self.generate_y_connected_components_tasks()
                self.generate_z_connected_components_tasks()
                #
                # Step 7: Do all connected components
                #
                rh_logger.logger.report_event(
                    "Making the AllConnectedComponentsTask")
                self.generate_all_connected_components_task()
                #
                # Step 8: Rewrite the segmentation
                #
                rh_logger.logger.report_event("Making volume relabeling task")
                self.generate_volume_relabeling_task()
                #
                # Step 9: Reblock the membrane prediction
                #
                rh_logger.logger.report_event(
                    "Making the membrane reblocking task")
                self.generate_membrane_block_task()
                #
                # Step 10: Download the ground truth
                #
                rh_logger.logger.report_event(
                    "Making the butterfly task to download the ground truth")
                self.generate_ground_truth_task()
                #
                # Step 11: Do the Neuroproof training
                #
                rh_logger.logger.report_event(
                    "Making the Neuroproof learning task")
                self.generate_neuroproof_learn_task()
                rh_logger.logger.report_event("Finished making tasks")
                self.requirements_computed = True
            except:
                rh_logger.logger.report_exception()
                raise
    
    def requires(self):
        self.compute_requirements()
        yield self.neuroproof_learn_task
    
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
        # There are n-1 overlaps for n blocks
        #
        self.n_x = int((self.useable_width-1-self.block_xy_overlap) /
                       (self.block_width - self.block_xy_overlap)) + 1
        self.n_y = int((self.useable_height-1-self.block_xy_overlap) / 
                       (self.block_height - self.block_xy_overlap)) + 1
        self.n_z = int((self.useable_depth-1-self.block_z_overlap) / 
                       (self.block_depth-self.block_z_overlap)) + 1
        self.xs = np.linspace(
            self.x0, self.x1, self.n_x, endpoint=False).astype(int)
        self.xe = np.zeros(self.xs.shape, int)
        self.xe[:-1] = self.xs[1:] + self.block_xy_overlap
        self.xe[-1] = self.x1
        self.ys = np.linspace(
            self.y0, self.y1, self.n_y, endpoint=False).astype(int)
        self.ye = np.zeros(self.ys.shape, int)
        self.ye[:-1] = self.ys[1:] + self.block_xy_overlap
        self.ye[-1] = self.y1
        self.zs = np.linspace(
            self.z0, self.z1, self.n_z, endpoint=False).astype(int)
        self.ze = np.zeros(self.zs.shape, int)
        self.ze[:-1] = self.zs[1:] + self.block_z_overlap
        self.ze[-1] = self.z1

    def generate_butterfly_tasks(self):
        '''Get volumes padded for CNN'''
        self.butterfly_tasks = np.zeros((self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            z0, z1 = self.zs[zi] - self.nn_z_pad, self.ze[zi] + self.nn_z_pad
            for yi in range(self.n_y):
                y0 = self.ys[yi] - self.nn_y_pad
                y1 = self.ye[yi] + self.nn_y_pad
                for xi in range(self.n_x):
                    x0 = self.xs[xi] - self.nn_x_pad
                    x1 = self.xe[xi] + self.nn_x_pad
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
                        border_width=self.block_xy_overlap,
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
    
    def generate_x_connected_components_tasks(self):
        '''Get connected components between adjacent blocks in the x direction'''
        self.x_connected_components_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x-1), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x-1):
                    wtask0 = self.watershed_tasks[zi, yi, xi]
                    wtask1 = self.watershed_tasks[zi, yi, xi+1]
                    #
                    # Get the concordance volume
                    #
                    x0 = int((wtask0.volume.x1 + wtask1.volume.x) / 2)
                    x1 = x0+1
                    y0 = max(wtask0.volume.y, wtask1.volume.y)
                    y1 = min(wtask0.volume.y1, wtask1.volume.y1)
                    z0 = max(wtask0.volume.z, wtask1.volume.z)
                    z1 = min(wtask0.volume.z1, wtask1.volume.z1)                    
                    cctask = self.factory.gen_connected_components_task(
                        volume1=wtask0.volume, 
                        location1=wtask0.output_location,
                        volume2=wtask1.volume,
                        location2=wtask1.output_location,
                        overlap_volume=Volume(x0, y0, z0,
                                              x1-x0, y1-y0, z1-z0),
                        output_location=self.get_connected_components_location(
                            x0, x1, y0, y1, z0, z1))
                    self.x_connected_components_tasks[zi, yi, xi] = cctask
                    cctask.set_requirement(wtask0)
                    cctask.set_requirement(wtask1)

    def generate_y_connected_components_tasks(self):
        '''Get connected components between adjacent blocks in the y direction'''
        self.y_connected_components_tasks = \
            np.zeros((self.n_z, self.n_y-1, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y-1):
                for xi in range(self.n_x):
                    wtask0 = self.watershed_tasks[zi, yi, xi]
                    wtask1 = self.watershed_tasks[zi, yi+1, xi]
                    #
                    # Get the concordance volume
                    #
                    x0 = max(wtask0.volume.x, wtask1.volume.x)
                    x1 = min(wtask0.volume.x1, wtask1.volume.x1)
                    y0 = int((wtask0.volume.y1 + wtask1.volume.y) / 2)
                    y1 = y0+1
                    z0 = max(wtask0.volume.z, wtask1.volume.z)
                    z1 = min(wtask0.volume.z1, wtask1.volume.z1)                    
                    cctask = self.factory.gen_connected_components_task(
                        volume1=wtask0.volume, 
                        location1=wtask0.output_location,
                        volume2=wtask1.volume,
                        location2=wtask1.output_location,
                        overlap_volume=Volume(x0, y0, z0,
                                              x1-x0, y1-y0, z1-z0),
                        output_location=self.get_connected_components_location(
                            x0, x1, y0, y1, z0, z1))
                    self.y_connected_components_tasks[zi, yi, xi] = cctask
                    cctask.set_requirement(wtask0)
                    cctask.set_requirement(wtask1)

    def generate_z_connected_components_tasks(self):
        '''Get connected components between adjacent blocks in the z direction'''
        self.z_connected_components_tasks = \
            np.zeros((self.n_z-1, self.n_y, self.n_x), object)
        for zi in range(self.n_z-1):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    wtask0 = self.watershed_tasks[zi, yi, xi]
                    wtask1 = self.watershed_tasks[zi+1, yi, xi]
                    #
                    # Get the concordance volume
                    #
                    x0 = max(wtask0.volume.x, wtask1.volume.x)
                    x1 = min(wtask0.volume.x1, wtask1.volume.x1)
                    y0 = max(wtask0.volume.y, wtask1.volume.y)
                    y1 = min(wtask0.volume.y1, wtask1.volume.y1)                    
                    z0 = int((wtask0.volume.z1 + wtask1.volume.z) / 2)
                    z1 = z0+1
                    cctask = self.factory.gen_connected_components_task(
                        volume1=wtask0.volume, 
                        location1=wtask0.output_location,
                        volume2=wtask1.volume,
                        location2=wtask1.output_location,
                        overlap_volume=Volume(x0, y0, z0,
                                              x1-x0, y1-y0, z1-z0),
                        output_location=self.get_connected_components_location(
                            x0, x1, y0, y1, z0, z1))
                    self.z_connected_components_tasks[zi, yi, xi] = cctask
                    cctask.set_requirement(wtask0)
                    cctask.set_requirement(wtask1)

    def generate_all_connected_components_task(self):
        '''Generate the task that makes the global segmentation mapping'''
        
        all_connected_components_tasks = \
            self.x_connected_components_tasks.flatten().tolist() +\
            self.y_connected_components_tasks.flatten().tolist() +\
            self.z_connected_components_tasks.flatten().tolist()
        input_locations = map(lambda task: task.output_location, 
                              all_connected_components_tasks)
        output_location = self.get_all_connected_components_location()
        self.all_connected_components_task = \
            self.factory.gen_all_connected_components_task(
                input_locations, output_location)
        for task in all_connected_components_tasks:
            self.all_connected_components_task.set_requirement(task)
    
    def generate_volume_relabeling_task(self):
        '''Make a global segmentation, relabeling all block segmentations'''
        inputs = []
        for ds in self.watershed_tasks.flatten().tolist():
            output = ds.output()
            inputs.append(dict(volume=output.volume,
                               location=output.dataset_location))
        relabeling_location = self.all_connected_components_task.output_location
        self.volume_relabeling_task =\
            self.factory.gen_volume_relabeling_task(
                input_volumes=inputs,
                relabeling_location=relabeling_location,
                output_volume=self.volume,
                output_location=self.get_segmentation_location())
        self.volume_relabeling_task.set_requirement(
            self.all_connected_components_task)

    def generate_membrane_block_task(self):
        '''Reblock the membrane prediction'''
        inputs = []
        classifier_tasks = self.classifier_tasks.flatten().tolist()
        for ds in classifier_tasks:
            output = ds.output()
            inputs.append(dict(volume=output.volume,
                               location=output.dataset_location))
        self.membrane_reblocking_task = self.factory.gen_block_task(
            inputs=inputs,
            output_volume=self.volume,
            output_location=self.get_global_prediction_location())
        for task in classifier_tasks:
            self.membrane_reblocking_task.set_requirement(task)
    
    def generate_ground_truth_task(self):
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
            location = dataset_location)
        
    def generate_neuroproof_learn_task(self):
        '''Learn, baby, learn'''
        prob_location = self.membrane_reblocking_task.output().dataset_location
        seg_location = self.volume_relabeling_task.output().dataset_location
        gt_location = self.ground_truth_task.output().dataset_location
        self.neuroproof_learn_task = \
            self.factory.gen_neuroproof_learn_task(
                volume=self.volume,
                prob_location=prob_location,
                seg_location=seg_location,
                gt_location=gt_location,
                output_location=self.output_location)
        self.neuroproof_learn_task.set_requirement(
            self.volume_relabeling_task)
        self.neuroproof_learn_task.set_requirement(
            self.membrane_reblocking_task)
        self.neuroproof_learn_task.set_requirement(
            self.ground_truth_task)

class NeuroproofPipelineTask(NeuroproofLearnPipelineTaskMixin, 
                             PipelineRunReportMixin,
                             luigi.Task):
    '''The Neuroproof pipeline task trains a Neuroproof classifier
    
    The pipeline performs the steps that result in the oversegmentation
    of the membrane probabilities. It then unites the volume blocks and
    sends the result into neuroproof_learn to train the classifier.
    
    To use, create a pickle of your classifier, point at a dataset with
    ground truth in Butterfly and the pipeline should do the rest.
    '''
    task_namespace = "ariadne_microns_pipeline"
