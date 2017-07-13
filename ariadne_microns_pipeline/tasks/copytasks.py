'''Copy various things

CopyStoragePlan copies one storage plan to another.
CopyFile copies one file to another
'''

import enum
import luigi
import numpy as np
import os
import rh_logger
import tifffile
import time

from .utilities import RunMixin, RequiresMixin, DatasetMixin
from ..targets import SrcVolumeTarget, DestVolumeReader
from ..parameters import Volume

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

class CopyStoragePlanTask(DatasetMixin,
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
        
class CopyLoadingPlanTask(DatasetMixin,
                          RunMixin,
                      RequiresMixin,
                      luigi.Task):
    '''A task to copy a loading plan to a storage plan'''

    task_namespace = "ariadne_microns_pipeline"
    src_loading_plan = luigi.Parameter(
        description="The source loading plan to be copied")
    offset = luigi.IntParameter(
        default=0,
        description="An offset to be added to every voxel value, e.g. "
            "to make segmentations unique across blocks")

    def input(self):
        for tgt in DestVolumeReader(self.src_loading_plan).get_source_targets():
            yield tgt

    def ariadne_run(self):
        src_tgt = DestVolumeReader(self.src_loading_plan)
        dest_tgt = self.output()
        data = src_tgt.imread()
        x0 = dest_tgt.volume.x - src_tgt.volume.x
        x1 = dest_tgt.volume.x1 - src_tgt.volume.x
        y0 = dest_tgt.volume.y - src_tgt.volume.y
        y1 = dest_tgt.volume.y1 - src_tgt.volume.y
        z0 = dest_tgt.volume.z - src_tgt.volume.z
        z1 = dest_tgt.volume.z1 - src_tgt.volume.z
        data = data[z0:z1, y0:y1, x0:x1]
        if self.offset != 0:
            data[data != 0] += data.dtype.type(self.offset)
        self.output().imwrite(data)

class CopyLoadingPlansTask(DatasetMixin,
                          RunMixin,
                          RequiresMixin,
                      luigi.Task):
    '''A task to copy from multiple loading plans to a storage plan'''

    task_namespace = "ariadne_microns_pipeline"
    src_loading_plans = luigi.ListParameter(
        description="The source loading plans to be copied")

    def input(self):
        for src_loading_plan in self.src_loading_plans:
            for tgt in DestVolumeReader(src_loading_plan).get_source_targets():
                yield tgt

    def ariadne_run(self):
        dest_tgt = self.output()
        assert isinstance(dest_tgt, SrcVolumeTarget)
        dest_volume = dest_tgt.volume
        data = np.zeros((dest_tgt.volume.depth,
                         dest_tgt.volume.height,
                         dest_tgt.volume.width),
                        dest_tgt.dtype)
        for src_loading_plan in self.src_loading_plans:
            src_tgt = DestVolumeReader(src_loading_plan)
            src_volume = src_tgt.volume
            assert isinstance(src_volume, Volume)
            if not src_volume.is_overlapping(dest_volume):
                continue
            overlap_volume = src_volume.get_overlapping_region(dest_volume)
            idata = src_tgt.imread()
            data[overlap_volume.z - dest_volume.z:
                 overlap_volume.z1 - dest_volume.z,
                 overlap_volume.y - dest_volume.y:
                 overlap_volume.y1 - dest_volume.y,
                 overlap_volume.x - dest_volume.x:
                 overlap_volume.x1 - dest_volume.x] = idata[
                     overlap_volume.z - src_volume.z:
                     overlap_volume.z1 - src_volume.z,
                     overlap_volume.y - src_volume.y:
                     overlap_volume.y1 - src_volume.y,
                     overlap_volume.x - src_volume.x:
                     overlap_volume.x1 - src_volume_x1]
        self.output().imwrite(data)

class AggregateOperation(enum.Enum):
    '''The aggregation operation for combining multiple data sources'''
    
    SUM=1,
    PRODUCT=2,
    MEAN=3,
    MEDIAN=4,
    MIN=5,
    MAX=6

class AggregateLoadingPlansTask(DatasetMixin,
                                  RunMixin,
                                  RequiresMixin,
                                  luigi.Task):
    '''This task aggregates the input loading plans to produce an output
    
    Any of the aggregate operations in AggregateOperation (e.g.
    AggregateOperation.MEAN) can be used to combine the values at each voxel
    of the inputs to produce the output.
    '''
    task_namespace = "ariadne_microns_pipeline"
    
    loading_plan_paths = luigi.ListParameter(
        description="The loading plans of the volumes to aggregate")
    operation = luigi.EnumParameter(
        enum=AggregateOperation,
        description="The operation to use to aggregate, e.g. MEAN")
    
    def input(self):
        for loading_plan in self.loading_plan_paths:
            for tgt in DestVolumeReader(loading_plan).get_source_targets():
                yield tgt
    
    def ariadne_run(self):
        loading_plans = [DestVolumeReader(_) for _ in self.loading_plan_paths]
        storage_plan = self.output()
        volume = storage_plan.volume
        dtype = storage_plan.dtype
        if dtype == np.uint8 and self.operation == AggregateOperation.MEAN:
            #
            # Temporarily use 16 bits for the summation of the mean
            #
            dtype = np.uint16
        if self.operation  == AggregateOperation.MEDIAN:
            #
            # Median isn't commutative, so all must be loaded
            #
            data = [_.imread().flatten() for _ in loading_plans]
            result = np.median(data, axis=0).reshape(
                volume.depth, volume.height, volume.width)
        else:
            if self.operation == AggregateOperation.MAX:
                op = np.max
            elif self.operation == AggregateOperation.MEAN:
                op = np.sum
            elif self.operation == AggregateOperation.MIN:
                op = np.min
            elif self.operation == AggregateOperation.PRODUCT:
                op = np.prod
            elif self.operation == AggregateOperation.SUM:
                op = np.sum
            result = loading_plans[0].imread().astype(dtype)
            for loading_plan in loading_plans[1:]:
                result = op([result, loading_plan.imread().astype(dtype)],
                            axis=0)
            if self.operation == AggregateOperation.MEAN:
                result = result / len(loading_plans)
        storage_plan.imwrite(result)    
    

class DeleteStoragePlan(RunMixin,
                        RequiresMixin,
                        luigi.Task):
    '''a task to delete a storage plan's .tif files'''
    
    task_namespace = "ariadne_microns_pipeline"
    
    dependency_outputs = luigi.ListParameter(
        default=[],
        description="The outputs of this task's dependencies. The task "
        "requests these as inputs so that all of them must be present "
        "before the storage plan is deleted.")
    storage_plan_path = luigi.Parameter(
        description="Storage plan to delete")
    
    def input(self):
        yield SrcVolumeTarget(self.storage_plan_path)
        for dependency_output in self.dependency_outputs:
            yield luigi.LocalTarget(dependency_output)
    
    def output(self):
        return luigi.LocalTarget(
            SrcVolumeTarget.storage_plan_path_to_deleted_file(
                self.storage_plan_path))
    
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

class ChimericSegmentationTask(DatasetMixin,
                               RunMixin,
                               RequiresMixin,
                               luigi.Task):
    '''A task to take two abutting segmentations and join them to form a third
    
    The third segmentation is numbered somewhat arbitrarily - the second
    half's IDs are offset by the maximum ID in the first.
    '''
    task_namespace = "ariadne_microns_pipeline"
    
    loading_plan1_path = luigi.Parameter(
        description="The path to the first loading plan")
    loading_plan2_path = luigi.Parameter(
        description="The path to the second loading plan")
    
    def input(self):
        for tgt in DestVolumeReader(self.loading_plan1_path)\
            .get_source_targets():
            yield tgt
        for tgt in DestVolumeReader(self.loading_plan2_path)\
            .get_source_targets():
            yield tgt
    
    def ariadne_run(self):
        tgt = self.output()
        result = np.zeros(
            (tgt.volume.depth, tgt.volume.height, tgt.volume.width),
            tgt.dtype)
        seg1_tgt = DestVolumeReader(self.loading_plan1_path)
        seg1_volume = seg1_tgt.volume
        seg1 = seg1_tgt.imread()
        result[seg1_volume.z - tgt.volume.z:seg1_volume.z1 - tgt.volume.z,
               seg1_volume.y - tgt.volume.y:seg1_volume.y1 - tgt.volume.y,
               seg1_volume.x - tgt.volume.x:seg1_volume.x1 - tgt.volume.x] = seg1
        seg2_tgt = DestVolumeReader(self.loading_plan2_path)
        seg2_volume = seg2_tgt.volume
        seg2 = seg2_tgt.imread()
        seg2[seg2 != 0] += np.max(seg1)
        result[seg2_volume.z - tgt.volume.z:seg2_volume.z1 - tgt.volume.z,
               seg2_volume.y - tgt.volume.y:seg2_volume.y1 - tgt.volume.y,
               seg2_volume.x - tgt.volume.x:seg2_volume.x1 - tgt.volume.x] = seg2
        tgt.imwrite(result)
        
        