'''Copy various things

CopyStoragePlan copies one storage plan to another.
CopyFile copies one file to another
'''

import luigi
import os
import rh_logger
import tifffile
import time

from .utilities import RunMixin, RequiresMixin, DatasetMixin
from ..targets import SrcVolumeTarget, DestVolumeReader

class CopyFileTask(
    RunMixin,
    RequiresMixin,
    luigi.Task):
    '''A task to copy a file to a new destination'''
    
    task_namespace = "ariadne_microns_pipeline"
    src_path = luigi.Parameter(
        description="Path to the source file")
    dest_path = luigi.Parameter(
        description="Path to the destination file")
    block_size = luigi.IntParameter(
        default=100*1000*1000,
        description="Size of a block of data being copied")
    
    def input(self):
        yield luigi.LocalTarget(self.src_path)
    
    def output(self):
        return luigi.LocalTarget(self.dest_path)
    
    def ariadne_run(self):
        with self.input().next().open("r") as fd_in:
            with self.output().open("w") as fd_out:
                while True:
                    buf = fd_in.read(self.block_size)
                    if len(buf) == 0:
                        break
                    fd_out.write(buf)

class CopyStoragePlan(DatasetMixin,
                      RunMixin,
                      RequiresMixin,
                      luigi.Task):
    '''A task to copy a storage plan'''
    
    task_namespace = "ariadne_microns_pipeline"
    src_storage_plan = luigi.Parameter(
        description="The source storage plan to be copied")
    
    def input(self):
        yield SrcVolumeTarget(self.src_storage_plan)
    
    def ariadne_run(self):
        data = self.input().next().imread()
        self.output().imwrite(data)
        
class DeleteStoragePlan(RunMixin,
                        RequiresMixin,
                        luigi.Task):
    '''a task to delete a storage plan's .tif files'''
    
    task_namespace = "ariadne_microns_pipeline"
    
    storage_plan_path = luigi.Parameter(
        description="Storage plan to delete")
    
    def input(self):
        yield SrcVolumeTarget(self.storage_plan_path)
    
    def output(self):
        return luigi.LocalTarget(self.storage_plan_path+".deleted")
    
    def ariadne_run(self):
        self.input().next().remove()
        with self.output().open("w") as fd:
            fd.write("So sorry.\n")

class BossShardingTaskMixin:
    loading_plan_path = luigi.Parameter(
        description="Location of the loading plan for this sharding")
    pattern = luigi.Parameter(
         description="Naming pattern for .png files. The path will be "
         "generated using pattern.format(x=x, y=y, z=z) where x, y and z "
         "are the origins of the tiles.")
    done_file = luigi.Parameter(
         description="Marker file written after task is done")
    
    def input(self):
        dest = DestVolumeReader(self.loading_plan_path)
        for tgt in dest.get_source_targets():
            yield tgt
    
    def output(self):
        return luigi.LocalTarget(self.done_file)

class BossShardingRunMixin:
    output_dtype = luigi.Parameter(
        default="uint32",
        description="The Numpy dtype of the output png files, e.g. \"uint32\"")
    compression = luigi.IntParameter(
        default=3,
        description="Amount of compression (0-10) to use on .tif files")
    
    def ariadne_run(self):
        '''Write the volume to .tif tiles'''
        src = DestVolumeReader(self.loading_plan_path)
        memory = src.imread()
        #
        # Write the planes
        #
        t0 = time.time()
        for idx, plane in enumerate(memory):
            path = self.pattern.format(
                x=src.volume.x,
                y=src.volume.y,
                z=src.volume.z + idx)
            if not os.path.isdir(os.path.dirname(path)):
                try:
                    os.makedirs(os.path.dirname(path))
                except:
                    # Race condition
                    pass
            tifffile.imsave(path, plane, compress=self.compression)
        rh_logger.logger.report_metric("BossShardingTask.write_time",
                                       time.time() - t0)
        with self.output().open("w") as fd:
            fd.write("done")

class BossShardingTask(
    BossShardingTaskMixin,
    BossShardingRunMixin,
    RunMixin,
    RequiresMixin,
    luigi.Task):
    '''A task to shard a segmentation for the BOSS
    
    The BOSS takes fixed-size, single plane tiles as input. This task
    takes a connectivity graph and prepares a stack of tiles.
    
    The input volume for this task should have the width and height of the
    tiles. The padding for the volume should be 1/2 of the padding used
    to make the segmentation volumes.
    '''
    task_namespace = "ariadne_microns_pipeline"

