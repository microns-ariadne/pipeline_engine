'''Synapse pipeline

This pipeline produces the synapse probability maps and the synapse
segmentations without processing the neurons.
'''

import luigi
import numpy as np
import os
import rh_logger

from ..tasks.factory import AMTaskFactory
from ..tasks.classify import ClassifyShimTask
from ..tasks.connected_components import FakeAllConnectedComponentsTask
from ..tasks.connected_components import JoiningMethod
from ..parameters import VolumeParameter, DatasetLocationParameter
from ..parameters import Volume, DatasetLocation, EMPTY_DATASET_LOCATION
from ..targets.factory import TargetFactory
from ..targets.classifier_target import PixelClassifierTarget

class SynapsePipelineTask(luigi.Task):
    '''The synapse pipeline task processes the synapses in a volume'''
    
    task_namespace = "ariadne_microns_pipeline"
    
    volume = VolumeParameter(description="The volume to process")
    output_location = luigi.Parameter(
        description="Directory for the segmentation .h5 file")
    temp_dirs = luigi.ListParameter(
        description="Locations for temp files")
    classifier_location= luigi.Parameter(
        description="Location for the classifier pickle file")
    experiment = luigi.Parameter(
        description="The Butterfly experiment that produced the dataset")
    sample = luigi.Parameter(
        description="The ID of the biological sample that was imaged")
    dataset = luigi.Parameter(
        description="The name of the volume that was imaged")
    channel = luigi.Parameter(
        default="raw",
        description="The name of the channel from which we take data")
    #
    # Optional parameters
    #
    butterfly_url = luigi.Parameter(
        default="http://localhost:2001/api",
        description="The URL for the butterfly server")
    block_width = luigi.IntParameter(
        default=2048,
        description="The width of a block")
    block_height = luigi.IntParameter(
        default=2048,
        description="The height of a block")
    block_depth = luigi.IntParameter(
        default=100,
        description="The depth of a block")
    xy_overlap = luigi.IntParameter(
        default=50,
        description="The amount of overlap between blocks in the x and y "
        "directions.")
    z_overlap = luigi.IntParameter(
        default=10,
        description="The amount of overlap between blocks in the z direction.")
    synapse_class_name = luigi.Parameter(
        default="synapse",
        description="The name of the synapse class in the classifier")
    #
    # FindSynapsesTask parameters
    #
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
        default=1)
    synapse_z_sigma = luigi.FloatParameter(
        description="Sigma for smoothing Gaussian for symapse segmentation "
                     "in the z direction.",
        default=.5)
    synapse_min_size_2d = luigi.IntParameter(
        default=100,
        description="Remove isolated synapse foreground in a plane if "
        "less than this # of pixels")
    synapse_max_size_2d = luigi.IntParameter(
        default=15000,
        description = "Remove large patches of mislabeled synapse in a plane "
        "that have an area greater than this")
    synapse_min_size_3d = luigi.IntParameter(
        default=500,
        description = "Minimum size in voxels of a synapse")
    min_synapse_depth = luigi.IntParameter(
        default=5,
        description="Minimum acceptable size of a synapse in the Z direction")
    synapse_threshold = luigi.FloatParameter(
        description="Threshold for synapse voxels vs background voxels",
        default=128.)
    #
    # connected components parameters
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
    
    
    def output(self):
        return luigi.LocalTarget(self.output_location+".done")
    
    def run(self):
        with self.output().open("w") as fd:
            fd.write("Done")
    
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
     
    def requires(self):
        self.compute_requirements()
        return self.requirements
    
    def compute_requirements(self):
        if hasattr(self, "requirements"):
            return
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
        
        self.task_factory = AMTaskFactory()
        rh_logger.logger.report_event(
            "Loading classifier from %s" % self.classifier_location)
        self.pixel_classifier = PixelClassifierTarget(self.classifier_location)
        self.compute_coordinates()
        self.compute_block_requirements()
        self.compute_stitching_requirements()
        
    def compute_coordinates(self):
        '''Compute the coordinates of the blocks'''
        self.n_x = int(np.ceil(float(self.volume.width) / self.block_width))
        self.n_y = int(np.ceil(float(self.volume.height) / self.block_height))
        self.n_z = int(np.ceil(float(self.volume.depth) / self.block_depth))
        x = np.linspace(self.volume.x, self.volume.x1, self.n_x+1).astype(int)
        self.xs = x[:-1]
        self.xe = x[1:]
        y = np.linspace(self.volume.y, self.volume.y1, self.n_y+1).astype(int)
        self.ys = y[:-1]
        self.ye = y[1:]
        z = np.linspace(self.volume.z, self.volume.z1, self.n_z+1).astype(int)
        self.zs = z[:-1]
        self.ze = z[1:]
    
    def compute_block_requirements(self):
        self.segmentation_tasks = \
            np.zeros((self.n_z, self.n_y, self.n_x), object)
        for zi in range(self.n_z):
            for yi in range(self.n_y):
                for xi in range(self.n_x):
                    self.segmentation_tasks[zi, yi, xi] = \
                        self.compute_block_requirement(xi, yi, zi)
    
    def compute_block_requirement(self, xi, yi, zi):
        x0 = self.xs[xi]
        x1 = self.xe[xi]
        y0 = self.ys[yi]
        y1 = self.ye[yi]
        z0 = self.zs[zi]
        z1 = self.ze[zi]
        
        # Account for overlap
        
        if x0 != self.volume.x:
            x0 -= self.xy_overlap
        if x1 != self.volume.x1:
            x1 += self.xy_overlap
        if y0 != self.volume.y:
            y0 -= self.xy_overlap
        if y1 != self.volume.y1:
            y1 += self.xy_overlap
        if z0 != self.volume.z:
            z0 -= self.z_overlap
        if z1 != self.volume.z:
            z1 += self.z_overlap
        
        volume = Volume(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
        
        #
        # Get the classifier input block coordinates
        #
        classifier_xpad = self.pixel_classifier.classifier.get_x_pad()
        classifier_ypad = self.pixel_classifier.classifier.get_y_pad()
        classifier_zpad = self.pixel_classifier.classifier.get_z_pad()
        
        cx0 = x0 - classifier_xpad
        cx1 = x1 + classifier_xpad
        cy0 = y0 - classifier_ypad
        cy1 = y1 + classifier_ypad
        cz0 = z0 - classifier_zpad
        cz1 = z1 + classifier_zpad
        
        classifier_input_volume = Volume(
            cx0, cy0, cz0, cx1 - cx0, cy1 - cy0, cz1 - cz0)
        #
        # The dataset locations
        #
        dl_butterfly = self.get_dataset_location(classifier_input_volume,
                                                 "image")
        dl_synapse = self.get_dataset_location(volume, "synapse-prediction")
        dl_segmentation = self.get_dataset_location(
            volume, "synapse-segmentation")
        #
        # Pipeline flow is Butterfly -> classifier -> shim -> find synapses
        #
        
        btask = self.task_factory.gen_get_volume_task(
            experiment=self.experiment,
            sample=self.sample,
            dataset=self.dataset,
            channel=self.channel,
            url=self.butterfly_url,
            volume=classifier_input_volume,
            location=dl_butterfly)
        
        paths = self.get_dirs(x0, y0, z0)
        ctask = self.task_factory.gen_classify_task(
            paths=paths, 
            datasets={self.synapse_class_name:"synapse-prediction"}, 
            pattern=self.get_pattern("synapse-prediction"),
            img_volume=btask.volume,
            img_location=btask.output().dataset_location,
            classifier_path=self.classifier_location)
        ctask.set_requirement(btask)
        
        shim_task = ClassifyShimTask.make_shim(
            classify_task=ctask,
            dataset_name="synapse-prediction")
        
        find_synapses_task = self.task_factory.gen_find_synapses_task(
            volume=volume,
            syn_location=shim_task.output().dataset_location,
            neuron_segmentation=EMPTY_DATASET_LOCATION,
            erosion_xy=self.synapse_xy_erosion,
            erosion_z=self.synapse_z_erosion,
            sigma_xy=self.synapse_xy_sigma,
            sigma_z=self.synapse_z_sigma,
            threshold=self.synapse_threshold,
            min_size_2d=self.synapse_min_size_2d,
            max_size_2d=self.synapse_max_size_2d,
            min_size_3d=self.synapse_min_size_3d,
            min_slice=self.min_synapse_depth,
            output_location=dl_segmentation)
        find_synapses_task.set_requirement(shim_task)
        return find_synapses_task
    
    def compute_stitching_requirements(self):
        '''Compute the tasks needed to stitch the blocks'''
        #
        # Pipeline is 
        # block -> 
        #      x-connections / y-connections / z-connections ->
        # all-connected-components ->
        # stitch segmentation
        #
        cc_tasks = []
        #
        # The x-blocks
        #
        for xi in range(self.n_x-1):
            for yi in range(self.n_y):
                for zi in range(self.n_z):
                    cc_tasks.append(
                        self.compute_x_connected_components_task(xi, yi, zi))
        #
        # The y-blocks
        #
        for yi in range(self.n_y-1):
            for xi in range(self.n_x):
                for yi in range(self.n_z):
                    cc_tasks.append(
                        self.compute_y_connected_components_task(xi, yi, zi))
        #
        # The z-blocks
        #
        for zi in range(self.n_z-1):
            for xi in range(self.n_x):
                for yi in range(self.n_y):
                    cc_tasks.append(
                        self.compute_z_connected_components_task(xi, yi, zi))
        #
        # The all-connected-components task
        #
        acc_location = os.path.join(
            self.get_dirs(self.xs[0], self.ys[0], self.zs[0])[0],
            "connectivity-graph.json")
        if len(cc_tasks) > 0:
            acc_task = self.task_factory.gen_all_connected_components_task(
                [_.output().path for _ in cc_tasks],
                acc_location)
            for task in cc_tasks:
                acc_task.set_requirement(task)
        else:
            # only one block - do a fake connected components
            seg_tgt = self.segmentation_tasks[0, 0, 0].output()
            acc_task = FakeAllConnectedComponentsTask(
                volume=seg_tgt.volume,
                location=seg_tgt.dataset_location,
                output_location=acc_location)
        for task in self.segmentation_tasks.flatten():
            acc_task.set_requirement(task)
        #
        # The stitching task
        #
        output_location = DatasetLocation(
            [self.output_location], 
            "synapse_segmentation",
            self.get_pattern("synapse_segmentation"))
        stask = self.task_factory.gen_stitch_segmentation_task(
             [], acc_task.output().path, self.volume, output_location)
        stask.set_requirement(acc_task)
        self.requirements = [stask]
    
    def configure_connected_components_task(self, task):
        task.joining_method = self.joining_method
        task.min_overlap_percent = self.min_percent_connected
        task.min_overlap_volume = self.min_overlap_volume
        task.max_poly_matches = self.max_poly_matches
        task.dont_join_orphans = self.dont_join_orphans
        task.orphan_min_overlap_ratio = self.orphan_min_overlap_ratio
        task.orphan_min_overlap_volume = self.orphan_min_overlap_volume
        
    def compute_x_connected_components_task(self, xi, yi, zi):
        task1 = self.segmentation_tasks[zi, yi, xi]
        tgt1 = task1.output()
        task2 = self.segmentation_tasks[zi, yi, xi+1]
        tgt2 = task2.output()
        y0 = max(tgt1.volume.y, tgt2.volume.y)
        y1 = min(tgt1.volume.y1, tgt2.volume.y1)
        z0 = max(tgt1.volume.z, tgt2.volume.z)
        z1 = min(tgt1.volume.z1, tgt2.volume.z1)
        overlap_volume = Volume(
            (tgt1.volume.x1 + tgt2.volume.x) / 2 - self.halo_size_xy / 2,
            y0, z0,
            self.halo_size_xy, y1-y0, z1-z0)
        
        output_location = os.path.join(
            self.get_dirs(tgt1.x, tgt1.y, tgt1.z)[0],
            "connected-components-x.json")
        cctask = self.task_factory.gen_connected_components_task(
            volume1=tgt1.volume,
            location1=tgt1.dataset_location,
            volume2=tgt2.volume,
            location2=tgt2.dataset_location,
            overlap_volume=overlap_volume,
            output_location=output_location)
        self.configure_connected_components_task(cctask)
        cctask.set_requirement(task1)
        cctask.set_requirement(task2)
        return cctask

    def compute_y_connected_components_task(self, xi, yi, zi):
        task1 = self.segmentation_tasks[zi, yi, xi]
        tgt1 = task1.output()
        task2 = self.segmentation_tasks[zi, yi+1, xi]
        tgt2 = task2.output()
        x0 = max(tgt1.volume.x, tgt2.volume.x)
        x1 = min(tgt1.volume.x1, tgt2.volume.x1)
        z0 = max(tgt1.volume.z, tgt2.volume.z)
        z1 = min(tgt1.volume.z1, tgt2.volume.z1)
        overlap_volume = Volume(
            x0, (tgt1.volume.y1 + tgt2.volume.y) / 2 - self.halo_size_xy / 2, z0,
            x1 - x0, self.halo_size_xy, z1-z0)
        
        output_location = os.path.join(
            self.get_dirs(tgt1.x, tgt1.y, tgt1.z)[0],
            "connected-components-x.json")
        cctask = self.task_factory.gen_connected_components_task(
            volume1=tgt1.volume,
            location1=tgt1.dataset_location,
            volume2=tgt2.volume,
            location2=tgt2.dataset_location,
            overlap_volume=overlap_volume,
            output_location=output_location)
        self.configure_connected_components_task(cctask)
        cctask.set_requirement(task1)
        cctask.set_requirement(task2)
        return cctask

    def compute_z_connected_components_task(self, xi, yi, zi):
        task1 = self.segmentation_tasks[zi, yi, xi]
        tgt1 = task1.output()
        task2 = self.segmentation_tasks[zi+1, yi, xi]
        tgt2 = task2.output()
        x0 = max(tgt1.volume.x, tgt2.volume.x)
        x1 = min(tgt1.volume.x1, tgt2.volume.x1)
        y0 = max(tgt1.volume.y, tgt2.volume.y)
        y1 = min(tgt1.volume.y1, tgt2.volume.y1)
        overlap_volume = Volume(
            x0, y0, (tgt1.volume.z1 + tgt2.volume.z) / 2 - self.halo_size_z / 2,
            x1 - x0, y1 - y0, self.halo_size_z)
        
        output_location = os.path.join(
            self.get_dirs(tgt1.x, tgt1.y, tgt1.z)[0],
            "connected-components-x.json")
        cctask = self.task_factory.gen_connected_components_task(
            volume1=tgt1.volume,
            location1=tgt1.dataset_location,
            volume2=tgt2.volume,
            location2=tgt2.dataset_location,
            overlap_volume=overlap_volume,
            output_location=output_location)
        self.configure_connected_components_task(cctask)
        cctask.set_requirement(task1)
        cctask.set_requirement(task2)
        return cctask
