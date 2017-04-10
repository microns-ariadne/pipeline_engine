'''Stitch a segmentation into a unitary form'''

import enum
import h5py
import json
import luigi
import numpy as np
import os
import Queue
import rh_logger
import multiprocessing
import time

from .utilities import RunMixin, RequiresMixin, SingleThreadedMixin, to_hashable
from ..parameters import VolumeParameter
from ..targets import DestVolumeReader

class Compression(enum.Enum):
    '''Compression types for HDF5'''
    NONE=0
    GZIP=1
    LZF=2

class StitchSegmentationTaskMixin:
    
    connected_components_location=luigi.Parameter(
        description="The output file from AllConnectedComponentsTask "
            "that gives the local <-> global label correspondences")
    output_volume=VolumeParameter(
        description="The volume of the output")
    output_location=luigi.Parameter(
        description="The location of the HDF5 file")
    
    def input(self):
        yield luigi.LocalTarget(self.connected_components_location)
        
    def output(self):
        return luigi.LocalTarget(self.output_location)

class StitchSegmentationRunMixin:
    
    dataset_name = luigi.Parameter(
        default="stack",
        description="The name for the dataset within the HDF5 file")
    xy_chunking = luigi.IntParameter(
        default=2048,
        description="The # of voxels in an HDF5 chunk in the x and y direction")
    z_chunking = luigi.IntParameter(
        default=4,
        description="The # of voxels in an HDF5 chunk in the z direction")
    x_padding = luigi.IntParameter(
        default=0,
        description="The amount of overlap between volumes in the x direction.")
    y_padding = luigi.IntParameter(
        default=0,
        description="The amount of overlap between volumes in the x direction.")
    z_padding = luigi.IntParameter(
        default=0,
        description="The amount of overlap between volumes in the z direction")
    compression = luigi.EnumParameter(
        enum=Compression,
        default=Compression.LZF,
        description="The type of compression for the HDF5 file")
    compression_level = luigi.IntParameter(
        default=4,
        description="For GZip, the level of compression (0 = lowest, 9 = most)")
    n_workers = luigi.IntParameter(
        default=4,
        description="Number of reader worker processes")
    
    def ariadne_run(self):
        output_tgt = self.output()
        chunks=(min(self.z_chunking, self.output_volume.depth),
                min(self.xy_chunking, self.output_volume.height),
                min(self.xy_chunking, self.output_volume.width))
        kwds = dict(chunks=chunks)
        if self.compression == Compression.GZIP:
            kwds["compression"] = "gzip"
            kwds["compression_opts"] = str(self.compression_level)
        elif self.compression == Compression.LZF:
            kwds["compression"] = "lzf"
        result = multiprocessing.Value("i")
        result.value = 0
        pool = multiprocessing.Pool(self.n_workers)
        with multiprocessing.Manager() as manager:
            queue = manager.Queue(4)
            with output_tgt.open("w") as fd:
                worker = multiprocessing.Process(
                    target = writer,
                    args = (queue, fd.name, self.dataset_name,
                        (self.output_volume.depth, 
                         self.output_volume.height, 
                         self.output_volume.width),
                        kwds, result))
                worker.start()
                x0 = self.output_volume.x
                y0 = self.output_volume.y
                z0 = self.output_volume.z
                x1 = self.output_volume.x1
                y1 = self.output_volume.y1
                z1 = self.output_volume.z1
                inputs = self.input()
                
                cc_location = inputs.next()
                t0 = time.time()
                cc = json.load(cc_location.open("r"),
                               object_hook = to_hashable)
                rh_logger.logger.report_metric(
                    "Component graph load time (sec)", time.time() - t0)
                #
                # This is a map of volume to local/global labelings
                #
                volume_map = dict(cc["volumes"])
                # Get input volumes from connected components
                #
                read_results = []
                for volume, location in cc["locations"]:
                    rr = pool.apply_async(
                        read_block, 
                        args=(location, x0, x1, y0, y1, z0, z1, volume_map, 
                              queue, self.x_padding, self.y_padding,
                              self.z_padding))
                    read_results.append(rr)
                pool.close()
                pool.join()
                for rr in read_results:
                    rr.get()
                if result.value < 0:
                    raise Exception("Writer process failed. See log for details")
                queue.put((None, None, None, None))
                if result.value < 0:
                    raise Exception("Writer process failed. See log for details")
                worker.join()

def read_block(location, x0, x1, y0, y1, z0, z1, volume_map, queue,
               x_padding, y_padding, z_padding):
    try:
        rh_logger.logger.start_process(
            "BlockReader", "Starting stitching reader", [])
    except:
        pass
    tgt = DestVolumeReader(location)
    if tgt.volume.x1 <= x0 or\
       tgt.volume.y1 <= y0 or\
       tgt.volume.z1 <= z0 or\
       tgt.volume.x >= x1 or \
       tgt.volume.y >= y1 or \
       tgt.volume.z >= z1:
        rh_logger.logger.report_event(
        "Skipping %d:%d, %d:%d, %d:%d" %
        (tgt.volume.x, tgt.volume.x1,
         tgt.volume.y, tgt.volume.y1,
         tgt.volume.z, tgt.volume.z1))
        rh_logger.logger.report_event(
            "Output volume: %d:%d %d:%d %d:%d" % (x0, x1, y0, y1, z0, z1))
        return
    tmp = np.array(
        volume_map[to_hashable(tgt.volume.to_dictionary())])
    mapping = np.zeros(np.max(tmp[:, 0]) + 1, np.uint32)
    mapping[tmp[:, 0]] = tmp[:, 1]
    seg = tgt.imread()
    t0 = time.time()
    seg = mapping[seg]
    rh_logger.logger.report_metric("Mapping time (sec)", time.time() - t0)
    
    x0a = max(x0, tgt.volume.x)
    if x0a != x0 and x_padding > 0:
        x0a += x_padding / 2 - 1
    x1a = min(x1, tgt.volume.x1)
    if x1a != x1 and x_padding > 0:
        x1a -= x_padding / 2 - 1
    y0a = max(y0, tgt.volume.y)
    if y0a != y0 and y_padding > 0:
        y0a += y_padding / 2 - 1
    y1a = min(y1, tgt.volume.y1)
    if y1a != y1 and y_padding > 0:
        y1a -= y_padding / 2 - 1
    z0a = max(z0, tgt.volume.z)
    if z0a != z0 and z_padding > 0:
        z0a += z_padding / 2 - 1
    z1a = min(z1, tgt.volume.z1)
    if z1a != z1 and z_padding > 0:
        z1a -= z_padding / 2 - 1
    if x1a < x0a or y1a < y0a or z1a < z0a:
        return
    queue.put((
        seg[z0a-tgt.volume.z:z1a-tgt.volume.z,
            y0a-tgt.volume.y:y1a-tgt.volume.y,
            x0a-tgt.volume.x:x1a-tgt.volume.x],
        x0a-x0, y0a-y0, z0a-z0))
    return (x1a - x0a, y1a-y0a, z1a-z0a)

def writer(queue, hdf_file, dataset_name, shape, create_dataset_kwds, result):
    '''The writer runs on its own thread, writing blocks from the queue.
    
    '''
    
    try:
        rh_logger.logger.start_process(
            "HDFWriter", "Starting stitching writer", [])
        rh_logger_started = True
    except:
        rh_logger.logger.report_event("Starting stitching writer")
        rh_logger_started = False
    try:
        t0 = time.time()
        with h5py.File(hdf_file, "w") as fd:
            rh_logger.logger.report_event(
                "Creating dataset with shape %s" % repr(shape))
            ds = fd.create_dataset(dataset_name, shape=shape, dtype=np.uint32, 
                                   **create_dataset_kwds)
            rh_logger.logger.report_metric(
                "HDF5 dataset creation (sec)", time.time() - t0)
            while True:
                block, x, y, z = queue.get()
                if block is None:
                    break
                t0 = time.time()
                ds[z:z+block.shape[0],
                   y:y+block.shape[1],
                   x:x+block.shape[2]] = block
                rh_logger.logger.report_metric("HDF5 write time (sec)", 
                                               time.time() - t0)
    except:
        result.value = -1
        rh_logger.logger.report_exception()
    if rh_logger_started:
        rh_logger.logger.end_process("Stitching writer exiting")

class StitchSegmentationTask(StitchSegmentationTaskMixin,
                             StitchSegmentationRunMixin,
                             RunMixin,
                             RequiresMixin,
                             SingleThreadedMixin,
                             luigi.Task):
    '''Stitch a segmentation together from blocks
    
    Create an HDF5 file with the stitched segmentation from many blocks.
    '''
    
    task_namespace = "ariadne_microns_pipeline"