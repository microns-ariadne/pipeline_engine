'''The ClassifyPipeline classifies a volume, writing the result to disk

'''
import itertools
import json
import luigi
import numpy as np
import os

from ..parameters import VolumeParameter, Volume
from .pipeline import IMG_DATASET
from ..targets.butterfly_target import LocalButterflyChannelTarget
from ..targets.volume_target import write_loading_plan, write_storage_plan
from ..targets.volume_target import DestVolumeReader
from ..tasks.factory import AMTaskFactory
from ..tasks.classify import ClassifyShimTask
from ..targets.classifier_target import PixelClassifierTarget
from ..volumedb import VolumeDB, Persistence, UINT8
import rh_logger

BUTTERFLY_PRIORITY = 0
BOSS_PRIORITY = 1
CLASSIFIER_PRIORITY = 2

class ClassifyPipelineTask(luigi.Task):
    task_namespace='ariadne_microns_pipeline'
    volume = VolumeParameter(
        description="The volume to classify")
    classifier = luigi.Parameter(
        description="The path to the classifier")
    temp_directory = luigi.Parameter(
        description="The path to the location for intermediate files")
    boss_directory = luigi.Parameter(
        description="The path to the BOSS files for output")
    boss_tile_width = luigi.IntParameter(
        description="Width of BOSS tiles",
        default=1024)
    boss_tile_height = luigi.IntParameter(
         description="Height of each BOSS tile",
         default=1024)
    boss_stack_depth = luigi.IntParameter(
         description="The number of z-slices processed by a BOSS task",
         default=64)
    classifier_block_width = luigi.IntParameter(
         description="The X-width in voxels of a block processed by "
        "the classifier",
        default=1024)
    classifier_block_height = luigi.IntParameter(
        description="The Y-height in voxels of a block processed by"
        " the classifier",
        default=1024)
    classifier_block_depth = luigi.IntParameter(
        description="The Z-depth in voxels of a block processed by the "
        "classifier",
        default=64)
    butterfly_index_file = luigi.Parameter(
        description="Pointer to the local butterfly index file giving the "
        "location of the image data")
    environment_id = luigi.Parameter(
        default="default",
        description="Name of the IPC worker target environment for "
        "the classifier.")
    pattern = luigi.Parameter(
        default="{dataset_name:s}/{x:09d}/{y:09d}/{z:09d}/{{z:09d}}.tif",
        description="The pattern for .tif files. This will be passed through "
        "str.format twice, once with the block coordinates and once with "
        "the coordinates of the individual slices. The first call has "
        "the parameters, format(x=x, y=y, z=z, row=row, column=column) where "
        "row and column are the row and column of the tile being written.")
    boss_tile_path = luigi.Parameter(
        description="Name of the file that has the dictionary of BOSS tiles")
    
    def classifier_target(self):
        '''The PixelClassifierTarget for the classifier'''
        return PixelClassifierTarget(self.classifier)
    
    def init_db(self):
        self.volume_db_url = "sqlite:///%s/volume.db" % self.temp_directory
        rh_logger.logger.report_event(
            "Creating volume DB at url %s" % self.volume_db_url)
        self.volume_db = VolumeDB(self.volume_db_url, "w")
        self.volume_db.set_target_dir(self.boss_directory) # unused but needed
        self.volume_db.set_temp_dir(self.temp_directory)
        ct = self.classifier_target()
        self.volume_db.register_dataset_type(
            IMG_DATASET, Persistence.Temporary, UINT8)
        for channel_name in ct.classifier.get_class_names():
            self.volume_db.register_dataset_type(
                channel_name, Persistence.Temporary, UINT8)
    
    def compute_task_priority(self, volume, base_priority):
        '''Get the priority of a task based on its position in the volume
        
        The idea here is to prioritize tasks closest to the origin in order
        to get contiguous tasks to be scheduled consecutively or concurrently.
        
        :param volume: the volume that the task operates on
        :param base_priority: a baseline priority multiplier to proiritize
                              the task by its type or category
        '''
        dx = volume.x - self.volume.x
        dy = volume.y - self.volume.y
        dz = volume.z - self.volume.z
        d = int(np.sqrt(dx*dx + dy*dy + dz * dz))
        return self.max_task_priority * (base_priority+1) - d
    
    @property
    def max_task_priority(self):
        '''The maximum priority that would be returned by compute_task_priority
        
        '''
        dx = self.volume.x1 - self.volume.x
        dy = self.volume.y1 - self.volume.y
        dz = self.volume.z1 - self.volume.z
        d = int(np.sqrt(dx*dx + dy*dy + dz * dz))
        return ((d+9999) / 10000) * 10000
    
    def compute_extents(self):
        butterfly = LocalButterflyChannelTarget(
           self.butterfly_index_file)
        #
        # Find the volume that we can classify
        #
        self.x0 = self.volume.x
        self.y0 = self.volume.y
        self.z0 = self.volume.z
        self.x1 = self.volume.x1
        self.y1 = self.volume.y1
        self.z1 = self.volume.z1
        ct = self.classifier_target()
        classifier = ct.classifier
        self.nn_x_pad = ct.x_pad
        self.nn_y_pad = ct.y_pad
        self.nn_z_pad = ct.z_pad
        self.x0 = max(ct.x_pad, self.volume.x)
        self.y0 = max(ct.y_pad, self.volume.y)
        self.z0 = max(ct.z_pad, self.volume.z)
        self.x1 = min(butterfly.x_extent - ct.x_pad, self.x1)
        self.y1 = min(butterfly.y_extent - ct.y_pad, self.y1)
        self.z1 = min(butterfly.z_extent - ct.z_pad, self.z1)
        #
        # Find the BOSS tiling of that volume.
        #
        self.boss_c0 = self.x0 / self.boss_tile_width
        self.boss_c1 = \
            (self.x1 + self.boss_tile_width - 1) / self.boss_tile_width
        self.boss_r0 = self.y0 / self.boss_tile_height
        self.boss_r1 = \
            (self.y1 + self.boss_tile_height - 1) / self.boss_tile_height
        self.boss_nx = self.boss_c1 - self.boss_c0
        self.boss_ny = self.boss_r1 - self.boss_r0
        self.boss_nz = (self.z1 - self.z0 + self.boss_stack_depth - 1 ) /\
            self.boss_stack_depth
        self.boss_x0 = self.boss_c0 * self.boss_tile_width
        self.boss_x1 = self.boss_c1 * self.boss_tile_width
        self.boss_y0 = self.boss_r0 * self.boss_tile_height
        self.boss_y1 = self.boss_r1 * self.boss_tile_height
        self.boss_xs = np.array(
            [_ * self.boss_tile_width 
             for _ in range(self.boss_c0, self.boss_c1)])
        self.boss_xe = self.boss_xs + self.boss_tile_width
        self.boss_ys = np.array(
            [_ * self.boss_tile_height 
             for _ in range(self.boss_c0, self.boss_c1)])
        self.boss_ye = self.boss_ys + self.boss_tile_height
        self.boss_zs = np.array(
            [self.z0 + self.boss_stack_depth * zidx 
             for zidx in range(self.boss_nz)])
        self.boss_ze = np.minimum(self.boss_zs + self.boss_stack_depth, self.z1)
        #
        # Find the classifier tiling of the volume
        #
        self.n_cx = (self.x1 - self.x0 + self.classifier_block_width - 1) / \
            self.classifier_block_width
        self.n_cy = (self.y1 - self.y0 + self.classifier_block_height - 1) / \
            self.classifier_block_height
        self.n_cz = (self.z1 - self.z0 + self.classifier_block_depth - 1) / \
            self.classifier_block_depth
        self.classifier_xso = np.array(
            [self.x0 + self.classifier_block_width * i 
             for i in range(self.n_cx)])
        self.classifier_xso[-1] = self.x1 - self.classifier_block_width
        self.classifier_yso = np.array(
            [self.y0 + self.classifier_block_height * i 
             for i in range(self.n_cy)])
        self.classifier_yso[-1] = self.y1 - self.classifier_block_height    
        self.classifier_zso = np.array(
            [self.z0 + self.classifier_block_depth * i 
             for i in range(self.n_cz)])
        self.classifier_zso[-1] = self.z1 - self.classifier_block_depth
        self.classifier_xeo = self.classifier_xso + self.classifier_block_width
        self.classifier_yeo = self.classifier_yso + self.classifier_block_height
        self.classifier_zeo = self.classifier_zso + self.classifier_block_depth
        self.classifier_xsi = self.classifier_xso - self.nn_x_pad
        self.classifier_xei = self.classifier_xeo + self.nn_x_pad
        self.classifier_ysi = self.classifier_yso - self.nn_y_pad
        self.classifier_yei = self.classifier_yeo + self.nn_y_pad
        self.classifier_zsi = self.classifier_zso - self.nn_z_pad
        self.classifier_zei = self.classifier_zeo + self.nn_z_pad

    def compute_requirements(self):
        try:
            rh_logger.logger.start_process("ClassifyPipeline", "starting", [])
        except:
            pass
        try:
            if not os.path.exists(self.temp_directory):
                os.makedirs(self.temp_directory)
            self.tasks = []
            self.datasets = {}
            self.requirements = []
            self.init_db()
            self.compute_extents()
            self.factory = AMTaskFactory(self.volume_db_url,
                                         self.volume_db)
            self.generate_butterfly_tasks()
            self.generate_classifier_tasks()
            self.generate_boss_tasks()
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
            dependentd = dict([(_, [] ) for _ in self.datasets])
            for task in self.tasks:
                for tgt in task.input():
                    path = tgt.path
                    if path in self.datasets:
                        task.set_requirement(self.datasets[path])
                        dependentd[path].append(task)
            
            return self.requirements
        except:
            rh_logger.logger.report_exception()
            raise
    
    def generate_butterfly_tasks(self):
        for xi, yi, zi in itertools.product(range(self.n_cx),
                                         range(self.n_cy),
                                         range(self.n_cz)):
            volume = Volume(self.classifier_xsi[xi], 
                            self.classifier_ysi[yi],
                            self.classifier_zsi[zi],
                            self.classifier_xei[xi] - self.classifier_xsi[xi],
                            self.classifier_yei[yi] - self.classifier_ysi[yi],
                            self.classifier_zei[zi] - self.classifier_zsi[zi])
            priority = self.compute_task_priority(volume, BUTTERFLY_PRIORITY)
            task = self.factory.gen_local_butterfly_task(
                self.butterfly_index_file,
                volume,
                IMG_DATASET)
            task.priority = priority
            self.datasets[task.output().path] = task
            self.tasks.append(task)
    
    def generate_classifier_tasks(self):
        classifier = self.classifier_target().classifier
        datasets = dict([(_,_) for _ in classifier.get_class_names()])
        for xi, yi, zi in itertools.product(range(self.n_cx),
                                            range(self.n_cy),
                                            range(self.n_cz)):
            input_volume = Volume(
                self.classifier_xsi[xi],
                self.classifier_ysi[yi],
                self.classifier_zsi[zi],
                self.classifier_xei[xi] - self.classifier_xsi[xi],
                self.classifier_yei[yi] - self.classifier_ysi[yi],
                self.classifier_zei[zi] - self.classifier_zsi[zi])
            output_volume = Volume(
                self.classifier_xso[xi],
                self.classifier_yso[yi],
                self.classifier_zso[zi],
                self.classifier_xeo[xi] - self.classifier_xso[xi],
                self.classifier_yeo[yi] - self.classifier_yso[yi],
                self.classifier_zeo[zi] - self.classifier_zso[zi])
            ctask = self.factory.gen_classify_task(
                datasets=datasets,
                img_volume = input_volume,
                output_volume=output_volume,
                dataset_name=IMG_DATASET,
                classifier_path = self.classifier,
                environment_id=self.environment_id)
            ctask.priority = self.compute_task_priority(input_volume,
                                                        CLASSIFIER_PRIORITY)
            self.tasks.append(ctask)
            for channel_name in classifier.get_class_names():
                shim_task = ClassifyShimTask.make_shim(
                    classify_task=ctask,
                    dataset_name=channel_name)    
                self.datasets[shim_task.output().path] = shim_task
                self.tasks.append(shim_task)
                shim_task.priority = ctask.priority
    
    def generate_boss_tasks(self):
        class_names = self.classifier_target().classifier.get_class_names()
        self.shard_tasks = \
            dict([(class_name, []) for class_name in class_names])
        for xi, yi, zi, dataset_name in itertools.product(range(self.boss_nx),
                                                          range(self.boss_ny),
                                                          range(self.boss_nz),
                                                          class_names):
            volume = Volume(self.boss_xs[xi],
                            self.boss_ys[yi],
                            self.boss_zs[zi],
                            self.boss_xe[xi] - self.boss_xs[xi],
                            self.boss_ye[yi] - self.boss_ys[yi],
                            self.boss_ze[zi] - self.boss_zs[zi])
            pattern = self.pattern.format(
                dataset_name=dataset_name,
                x=volume.x,
                y=volume.y,
                z=volume.z,
                column=self.boss_c0 + xi,
                row=self.boss_r0 + yi)
            done_file = os.path.join(self.temp_directory,
                                     "done_files",
                                     "%s_boss_sharding_task_%d_%d_%d.done" %
                                     (dataset_name, xi, yi, zi))
            
            task = self.factory.gen_boss_sharding_task(
                volume=volume,
                dataset_name=dataset_name,
                output_dtype="uint8",
                pattern=pattern,
                done_file=done_file)
            task.priority = self.compute_task_priority(volume, BOSS_PRIORITY)
            self.tasks.append(task)
            self.shard_tasks[dataset_name].append(task)
            self.requirements.append(task)
    
    def requires(self):
        if not hasattr(self, "requirements"):
            self.compute_requirements()
        return self.requirements
    
    def input(self):
        if not getattr(self, "requirements"):
            self.compute_requirements()
        for task in self.requirements:
            yield task.output()
    
    def output(self):
        return luigi.LocalTarget(self.boss_tile_path)
    
    def run(self):
        class_names = self.classifier_target().classifier.get_class_names()
        tiles = dict([(_, []) for _ in class_names])
        for dataset_name in self.shard_tasks:
            for task in self.shard_tasks[dataset_name]:
                pattern = task.pattern
                volume = DestVolumeReader(task.loading_plan_path).volume
                for z in range(volume.z, volume.z1):
                    tile = dict(column = volume.x / self.boss_tile_width,
                                row = volume.y / self.boss_tile_height,
                                z=z,
                                location = pattern.format(z=z))
                    tiles[dataset_name].append(tile)
        with self.output().open("w") as fd:
            json.dump(tiles, fd)