'''The BOSS pipeline shards a segmentation into .tif files for upload to th BOSS

'''

import itertools
import json
import luigi
import numpy as np
import os
import rh_logger
import sqlite3

from .pipeline import NP_DATASET, SYN_SEG_DATASET
from ..parameters import VolumeParameter, Volume, EMPTY_LOCATION
from ..tasks import CopyFileTask, AMTaskFactory
from ..tasks.utilities import RunMixin, RequiresMixin, DatasetMixin
from ..tasks import DeleteStoragePlan
from ..volumedb import VolumeDB, Persistence
from ..targets.volume_target import write_loading_plan, write_storage_plan

PRIORITY_RELABELING_TASK = 1
PRIORITY_SHARDING_TASK = 2
PRIORITY_DELETE_TASK = 3

class BossPipelineTaskMixin:
    
    ###########################
    #
    # File parameters
    #
    ###########################
    
    connectivity_graph_path = luigi.Parameter(
        description="Path to the connectivity graph describing the global "
        "segmentation")
    temp_dir = luigi.Parameter(
        description="Path to temporary storage")
    pattern = luigi.Parameter(
        default="{x:09d}/{y:09d}/{z:09d}/seg{{z:09d}}.tif",
        description="The pattern for .tif files. This will be passed through "
        "str.format twice, once with the block coordinates and once with "
        "the coordinates of the individual slices. The first call has "
        "the parameters, format(x=x, y=y, z=z, row=row, column=column) where "
        "row and column are the row and column of the tile being written.")
    boss_configuration_path = luigi.Parameter(
        description="Location for the BOSS configuration file "
        "(see https://github.com/jhuapl-boss/ingest-client/wiki/"
        "Creating-Ingest-Job-Configuration-Files)")
    volume_db_url  = luigi.Parameter(
        description="The location of the database for the VolumeDB",
        default="sqlite:///")
    collection = luigi.Parameter(
        description="The name of the experiment collection")
    experiment = luigi.Parameter(
        description="The name of the experiment volume being uploaded")
    channel = luigi.Parameter(
        description="The name of the segmentation channel")
    tile_database_path=luigi.Parameter(
        description="The path of the sqlite database to hold tile locations")
    tile_json_path = luigi.Parameter(
        description="The path to the json file holding tile locations")
    done_file_folder = luigi.Parameter(
        default="/tmp",
        description="A folder for holding the done files of the sharding "
        "tasks.")
    tile_datatype = luigi.Parameter(
        default="uint64",
        description="Datatype for segmentation voxels")
    
    ############################
    #
    # Block parameters
    #
    ############################
    
    volume = VolumeParameter(
        description="The volume extent being prepared for upload")
    tile_width = luigi.IntParameter(
        default=1024,
        description="The width of a tile")
    tile_height = luigi.IntParameter(
        default=1024,
        description="The height of a tile")
    stack_depth = luigi.IntParameter(
        default=100,
        description="The number of planes per task")
    x_pad = luigi.IntParameter(
        description="Amount to crop off of each Neuroproof block in X")
    y_pad = luigi.IntParameter(
        description="Amount to crop off of each Neuroproof block in Y")
    z_pad = luigi.IntParameter(
        description="Amount to crop off of each Neuroproof block in Z")
    
    #############################
    #
    # Internal running parameters
    #
    #############################
    
    bulk_insert_size = luigi.IntParameter(
        default=256,
        description="number of rows inserted in the database per bulk insert")
    
    def input(self):
        yield luigi.LocalTarget(self.connectivity_graph_path)
    
    def output(self):
        '''The BOSS configuration file is the last thing written'''
        return luigi.LocalTarget(self.boss_configuration_path)
    
    def requires(self):
        if not hasattr(self, "requirements"):
            self.compute_extents()
            self.requirements = self.compute_requirements()
        return self.requirements
    
    def compute_extents(self):
        '''Compute the number and size of blocks'''
        self.n_x = (self.volume.width + self.tile_width - 1) / \
            self.tile_width
        self.n_y = (self.volume.height + self.tile_height - 1) / \
            self.tile_height
        self.n_z = (self.volume.depth + self.stack_depth - 1) / \
            self.stack_depth
        self.xs = np.arange(self.n_x) * self.tile_width + self.volume.x
        self.xe = self.xs + self.tile_width
        self.ys = np.arange(self.n_y) * self.tile_height + self.volume.y
        self.ye = self.ys + self.tile_height
        self.zs = np.arange(self.n_z) * self.stack_depth + self.volume.z
        self.ze = np.minimum(self.zs + self.stack_depth, self.volume.z1)
    
    def compute_requirements(self):
        '''Return the tasks needed to make the shards'''
        try:
            rh_logger.logger.start_process("Boss pipeline", "starting", [])
        except:
            pass
        if not os.path.isdir(self.done_file_folder):
            os.makedirs(self.done_file_folder)
        #
        # Set up the task factory and volume DB
        #
        volume_db = VolumeDB(self.volume_db_url, "w")
        factory = AMTaskFactory(self.volume_db_url, volume_db)
        volume_db.set_temp_dir(self.temp_dir)
        volume_db.register_dataset_type(NP_DATASET, 
                                        Persistence.Temporary,
                                        self.tile_datatype)
        volume_db.register_dataset_type(SYN_SEG_DATASET,
                                        Persistence.Temporary,
                                        self.tile_datatype)
        #
        # Set up to copy the connectivity graph to local storage
        #
        connectivity_graph_path = os.path.join(self.temp_dir, 
                                                   "connectivity-graph.json")
        cg = json.load(open(self.connectivity_graph_path))
        cc_task = CopyFileTask(
            src_path = self.connectivity_graph_path,
            dest_path=connectivity_graph_path)
        #
        # Generate the source plan copy tasks
        #
        min_x = self.volume.x1
        min_y = self.volume.y1
        min_z = self.volume.z1
        max_x = self.volume.x
        max_y = self.volume.y
        max_z = self.volume.z
        volumes = []
        for volume, location in cg["locations"]:
            volume = Volume(**volume)
            if volume.overlaps(self.volume):
                volumes.append((volume, location))
                min_x = min(min_x, volume.x)
                max_x = max(max_x, volume.x1)
                min_y = min(min_y, volume.y)
                max_y = max(max_y, volume.y1)
                min_z = min(min_z, volume.z)
                max_z = max(max_z, volume.z1)
        rh_logger.logger.report_event("Generating relabeling tasks")
        relabeling_tasks_by_storage_plan = {}
        for volume, location in volumes:
            if volume.x == min_x:
                x0 = min_x
            else:
                x0 = volume.x + self.x_pad
            if volume.x1 == max_x:
                x1 = max_x
            else:
                x1 = volume.x1 - self.x_pad + 1
            if volume.y == min_y:
                y0 = min_y
            else:
                y0 = volume.y + self.y_pad
            if volume.y1 == max_y:
                y1 = max_y
            else:
                y1 = volume.y1 - self.y_pad + 1
            if volume.z == min_z:
                z0 = min_z
            else:
                z0 = volume.z + self.z_pad
            if volume.z1 == max_z:
                z1 = max_z
            else:
                z1 = volume.z1 - self.z_pad + 1
            volume = Volume(x0, y0, z0, x1-x0, y1-y0, z1-z0)
            if not volume.overlaps(self.volume):
                # Padding disqualified it
                continue
            task = factory.gen_storage_plan_relabeling_task(
                connectivity_graph_path, 
                volume, 
                location,
                dataset_name=NP_DATASET)
            task.priority = PRIORITY_RELABELING_TASK
            task.set_requirement(cc_task)
            done_file = task.output().path
            relabeling_tasks_by_storage_plan[done_file] = task
        
        rh_logger.logger.report_event("Generating sharding tasks")
        sharding_tasks = []
        for xi, yi, zi in itertools.product(range(self.n_x),
                                            range(self.n_y),
                                            range(self.n_z)):
            volume = Volume(self.xs[xi],
                            self.ys[yi],
                            self.zs[zi],
                            self.tile_width,
                            self.tile_height,
                            self.ze[zi] - self.zs[zi])
            pattern = self.pattern.format(
                x=volume.x,
                y=volume.y,
                z=volume.z,
                column=xi,
                row=yi)
            done_file = "boss_sharding_task_%d_%d_%d.done" % (xi, yi, zi)
            done_path = os.path.join(self.done_file_folder, done_file)
            task = factory.gen_boss_sharding_task(
                volume=volume, 
                dataset_name=NP_DATASET,
                output_dtype=self.tile_datatype,
                pattern=pattern,
                done_file=done_path)
            task.priority = PRIORITY_SHARDING_TASK
            sharding_tasks.append(task)
        #
        # Do the VolumeDB computation
        #
        rh_logger.logger.report_event("Computing load/store plans")
        volume_db.compute_subvolumes()
        #
        # Write the loading plans
        #
        rh_logger.logger.report_event("Writing loading plans")
        for loading_plan_id in volume_db.get_loading_plan_ids():
            loading_plan_path = volume_db.get_loading_plan_path(
                loading_plan_id)
            directory = os.path.dirname(loading_plan_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            write_loading_plan(loading_plan_path, volume_db, 
                              loading_plan_id)
        #
        # Write the storage plans
        #
        rh_logger.logger.report_event("Writing storage plans")
        for dataset_id in volume_db.get_dataset_ids():
            write_storage_plan(volume_db, dataset_id)
        #
        # Hook up dependencies.
        #
        tasks_by_storage_plan = {}
        for task in sharding_tasks:
            for tgt in task.input():
                path = tgt.path
                if path in relabeling_tasks_by_storage_plan:
                    task.set_requirement(relabeling_tasks_by_storage_plan[path])
                storage_plan = tgt.storage_plan_path
                if storage_plan not in tasks_by_storage_plan:
                    tasks_by_storage_plan[storage_plan] = []
                tasks_by_storage_plan[storage_plan].append(task)
            yield task
        
        for storage_plan in tasks_by_storage_plan:
            dependent_tasks = tasks_by_storage_plan[storage_plan]
            dependent_outputs = [_.output().path for _ in dependent_tasks]
            delete_task = DeleteStoragePlan(
                dependency_outputs=dependent_outputs,
                storage_plan_path=storage_plan)
            delete_task.priority = PRIORITY_DELETE_TASK
            map(delete_task.set_requirement, dependent_tasks)
            yield delete_task
    
    def run(self):
        '''At the end of it all, write the config and index files'''
        #
        # The SQL connection
        #
        connection = sqlite3.connect(self.tile_database_path)
        cursor = connection.cursor()
        #
        # Make the database schema
        #
        cursor.execute("drop table if exists tiles")
        cursor.execute("""
        create table if not exists
        tiles (
            column integer not null,
            row integer not null,
            z integer not null,
            location text not null,
            primary key (column, row, z)
        )
        """)
        #
        # The base JSON object for both the BOSS upload and the
        # tiles
        #
        base = dict(
            database=dict(
                collection=self.collection,
                experiment=self.experiment,
                channel=self.channel),
            ingest_job=dict(
                resolution=0,
                extent=dict(
                    x=[self.volume.x, self.volume.x1],
                    y=[self.volume.y, self.volume.y1],
                    z=[self.volume.z, self.volume.z1],
                    t=[0, 1]
                    ),
                tile_size=dict(
                    x=self.tile_width,
                    y=self.tile_height,
                    z=1,
                    t=1
                )
            )
        )
        #
        # The tile index: dictionaries of row, column, z and location
        #                 stored in a list.
        #
        tiles = []
        tile_index = base.copy()
        tile_index["tiles"] = tiles
        #
        # rows for the bulk insert
        #
        rows = []
        insert_stmt = \
            "insert into tiles (column, row, z, location) values (?,?,?,?)"
        #
        # Loop over all blocks
        #
        for xi, yi, zi in itertools.product(range(self.n_x),
                                                range(self.n_y),
                                                range(self.n_z)):
            pattern = self.pattern.format(
                x=self.xs[xi],
                y=self.ys[yi],
                z=self.zs[zi],
                column=xi,
                row=yi)
            for z in range(self.zs[zi], self.ze[zi]):
                location = pattern.format(z=z)
                tiles.append(dict(column=xi,
                                  row=yi,
                                  z=z,
                                  location=location))
                rows.append((xi, yi, z, location))
                if len(rows) >= self.bulk_insert_size:
                    cursor.executemany(insert_stmt, rows)
                    rows = []
        if len(rows) >= 0:
            cursor.executemany(insert_stmt, rows)
        connection.commit()
        cursor.close()
        connection.close()
        #
        # Write the JSON index
        #
        json.dump(tile_index, open(self.tile_json_path, "w"))
        #
        # Here's the BOSS ingest file
        #
        boss = base.copy()
        boss.update(
            {
                "schema": {
                  "name": "boss-v0.1-schema",
                  "validator": "BossValidatorV01"
                  },
              "client": {
                  "backend": {
                    "name": "boss",
                  "class": "BossBackend",
                  "host": "api.theboss.io",
                  "protocol": "https"
                  },
                "path_processor": {
                    "class": "ingestclient.plugins.pipeline_tiles.PipelineTilePathProcessor",
                  "params": {
                      "database": self.tile_database_path
                  }
                  },
                "tile_processor": {
                    "class": "ingestclient.plugins.pipeline_tiles.PipelineTileProcessor",
                  "params": {
                      "datatype": self.tile_datatype,
                      "filetype": "tif",
                      "width": self.volume.width,
                      "height": self.volume.height,
                      "depth": self.volume.depth,
                      "tile_width": self.tile_width,
                      "tile_height": self.tile_height
                  }
                }
              }
            })
        with self.output().open("w") as fd:
            json.dump(boss, fd)

class BossPipelineTask(BossPipelineTaskMixin,
                       luigi.Task):
    '''A task to create tiles from a segmentation for upload to the BOSS
    
    This pipeline does several things:
    
    * Creates rows and columns uniformly sized tiles of the global segmentation
    * Creates a database suitable for our custom ingest client tile and path
      processors (ingest.plugins.pipeline_tiles.PipelineTilePathProcessor and
      ingest.plugins.pipeline_tiles.PipelineTileProcessor)
    * Creates an index file that might be used by Butterfly to access the
      segmentation.
    * Creates a BOSS ingest configuration file.
    
    The format of the index file is similar to the the ingest configuration
    file (it shares the database, ingest_job and tile_size dictionaries)
    and has an additional keyword, "tiles" which holds a list of dictionaries
    which have attributes, "column" (the x tile index), "row" (the y tile index)
    and "z" (the z tile index) and "location" (the path to the .tif file
    containing the segmentation).
    '''
    
    task_namespace="ariadne_microns_pipeline"

