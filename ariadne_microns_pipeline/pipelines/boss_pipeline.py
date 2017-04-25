'''The BOSS pipeline shards a segmentation into .tif files for upload to th BOSS

'''

import itertools
import json
import luigi
import numpy as np
import os
import rh_logger
import sqlite3

from ..parameters import VolumeParameter, Volume, EMPTY_LOCATION
from ..tasks import BossShardingTask

class BossPipelineTaskMixin:
    
    ###########################
    #
    # File parameters
    #
    ###########################
    
    connectivity_graph_path = luigi.Parameter(
        description="Path to the connectivity graph describing the global "
        "segmentation")
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
        default="uint32",
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
            task = BossShardingTask(
                connectivity_graph_path=self.connectivity_graph_path,
                volume=volume,
                pattern=pattern,
                done_file=done_path,
                output_dtype=self.tile_datatype,
                x_pad=self.x_pad,
                y_pad=self.y_pad,
                z_pad=self.z_pad)
            yield task
    
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
                    )
                ),
            tile_size=dict(
                x=self.tile_width,
                y=self.tile_height,
                z=1,
                t=1
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
                    "class": "ingest.plugins.pipeline_tiles.PipelineTilePathProcessor",
                  "params": {
                      "database": self.tile_database_path
                  }
                  },
                "tile_processor": {
                    "class": "ingest.plugins.pipeline_tiles.PipelineTileProcessor",
                  "params": {
                      "datatype": self.tile_datatype,
                    "filetype": "tif"
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