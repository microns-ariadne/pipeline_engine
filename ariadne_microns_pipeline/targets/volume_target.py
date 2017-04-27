import luigi
import numpy as np
import json
import os
import rh_logger
import tifffile
import time

from ..parameters import Volume, EMPTY_DATASET_ID, EMPTY_LOADING_PLAN_ID
from ..volumedb import VolumeDB

# Compression = 0
#    Time per call = 1.04
#    Time per gVoxel = 31.04
#    Compression = 400.04 %
# Compression = 1
#    Time per call = 1.09
#    Time per gVoxel = 32.40
#    Compression = 11.97 %
# Compression = 2
#    Time per call = 1.06
#    Time per gVoxel = 31.69
#    Compression = 10.96 %
# Compression = 3
#   Time per call = 1.06
#    Time per gVoxel = 31.50
#    Compression = 10.07 %
# Compression = 4
#    Time per call = 2.72
#    Time per gVoxel = 81.01
#    Compression = 8.80 %
# Compression = 5
#    Time per call = 2.78
#    Time per gVoxel = 82.87
#    Compression = 8.04 %
# Compression = 6
#    Time per call = 3.16
#    Time per gVoxel = 93.96
#    Compression = 6.13 %
# Compression = 7
#    Time per call = 3.30
#    Time per gVoxel = 98.30
#    Compression = 6.04 %
# Compression = 8
#    Time per call = 5.02
#    Time per gVoxel = 149.50
#    Compression = 5.47 %
# Compression = 9
#    Time per call = 7.96
#    Time per gVoxel = 237.01
#    Compression = 5.38 %
'''The compression factor for the tiff file.'''
COMPRESSION = 3

def write_storage_plan(volume_db, dataset_id):
    '''Write the plan for storing a dataset to disk
    
    :param volume_db: the VolumeDB database containing the dataset record
    :param dataset_id: the dataset ID record.
    
    Plans get written to a filename generated by replacing ".done" with ".plan"
    for the dataset's ".done" file.
    '''
    assert isinstance(volume_db, VolumeDB)
    
    done_path = volume_db.get_dataset_path(dataset_id)
    path = os.path.splitext(done_path)[0] + ".plan"
    
    blocks = []
    for location, volume \
        in volume_db.get_subvolume_locations_by_dataset_id(dataset_id):
        blocks.append([volume.to_dictionary(), location])
    dtype = volume_db.get_dataset_dtype_by_dataset_id(dataset_id)
    volume = volume_db.get_dataset_volume(dataset_id)
    d = dict(dimensions=[volume.depth,
                         volume.height,
                         volume.width],
             x=volume.x,
             y=volume.y,
             z=volume.z,
             blocks=blocks,
             datatype=dtype,
             dataset_name=volume_db.get_dataset_name_by_dataset_id(dataset_id),
             dataset_id=dataset_id)
    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        try:
            # Believe it or not, have seen a race where two tasks try
            # to make the directory at the same time.
            os.makedirs(path_dir)
        except:
            # Failures other than the race will fail upon trying to create
            # the done file.
            rh_logger.logger.report_exception()
    with open(path, "w") as fd:
        json.dump(d, fd)

def write_simple_storage_plan(
    storage_plan_path, dataset_path, volume, dataset_name, dtype):
    '''Write a simple storage plan for storing a volume in a single TIF stack
    
    :param storage_plan_path: where to put the storage plan
    :param dataset_path: the name of the .tif file that should be created
    :param volume: an ariadne_microns_pipeline.parameters.Volume describing
                   the extent of the voxel data
    :param dataset_name: the name of the datatype, e.g. "image"
    :param dtype: a string representation of the Numpy dtype, e.g. "uint8"
    '''
    blocks = [(volume.to_dictionary(), dataset_path)]
    d = dict(dimensions=[volume.depth,
                         volume.height,
                         volume.width],
             x=volume.x,
             y=volume.y,
             z=volume.z,
             blocks=blocks,
             datatype=dtype,
             dataset_name=dataset_name,
             dataset_id=EMPTY_DATASET_ID)
    with open(storage_plan_path, "w") as fd:
        json.dump(d, fd)

class SrcVolumeTarget(luigi.LocalTarget):
    '''A volume target that can be used to write a volume
    
    '''
    
    def __init__(self, storage_plan_path, compression=COMPRESSION):
        '''Initialize the target with the pathnames and file name pattern

        :param storage_plan: the path to the storage plan file, e.g. as
        written by write_storage_plan()
        :param compression: the degree of compression for the .tif file.
        '''
        self.storage_plan_path = storage_plan_path
        self.compression=compression
        done_file = os.path.splitext(storage_plan_path)[0] + ".done"
        self.__volume = None
        super(SrcVolumeTarget, self).__init__(done_file)

    def __getstate__(self):
        return self.storage_plan_path
    
    def __setstate__(self, state):
        self.storage_plan_path = state
        done_file = os.path.splitext(storage_plan_path)[0] + ".done"
        super(VolumeTarget, self).__init__(done_file)
    
    @property
    def volume(self):
        '''The global volume covered by this target'''
        if self.__volume is None:
            with open(self.storage_plan_path, "r") as fd:
                d = json.load(fd)
                depth, height, width = d["dimensions"]
                self.__volume = Volume(d["x"], d["y"], d["z"],
                                       width, height, depth)
        return self.__volume
    
    @property
    def dataset_name(self):
        '''The name of the storage plan dataset'''
        with open(self.storage_plan_path, "r") as fd:
            d = json.load(fd)
        return d["dataset_name"]
    
    def create_directories(self):
        '''Create the subdirectories to host the .tif files'''
        with open(self.storage_plan_path, "r") as fd:
            d = json.load(fd)
        for subvolume, tif_path in d["blocks"]:
            tif_dir = os.path.dirname(tif_path)
            if not os.path.isdir(tif_dir):
                try:
                    os.makedirs(tif_dir)
                except:
                    #
                    # Have seen a race between this and another task
                    # to make the directory.
                    #
                    rh_logger.logger.report_exception()
    
    def remove(self):
        '''Remove the tif files and done file for a target'''
        with open(self.storage_plan_path, "r") as fd:
            d = json.load(fd)
        for subvolume, tif_path in d["blocks"]:
            try:
                os.remove(tif_path)
            except:
                rh_logger.logger.report_exception(
                    "Failed to remove " + tif_path)
        try:
            os.remove(self.path)
        except:
            rh_logger.logger.report_exception(
                "Failed to remove " + self.path)
        
    def finish_imwrite(self):
        '''Just copy the storage plan to the output done file destination'''
        with open(self.storage_plan_path, "r") as fd:
            d = json.load(fd)
            with self.open("w") as fd:
                json.dump(d, fd)        
    
    def imwrite(self, data):
        '''Write the data blocks to disk + the done file
        
        The .done file is a json-encoded dictionary with the following keys
        
        * dimensions - the dimensions of the volume in the order, 
          depth, height, width
        * x - the x offset of the origin of the volume
        * y - the y offset of the origin of the volume
        * z - the z offset of the origin of the volume
        * blocks - a list of two tuples. The first tuple is the volume
                   of the block encoded as a dictionary. The second element
                   of the tuple is the location of the .tif file holding
                   the block.
        '''
        t0 = time.time()
        with open(self.storage_plan_path, "r") as fd:
            d = json.load(fd)
        depth, height, width = d["dimensions"]
        x0 = d["x"]
        x1 = x0 + width
        y0 = d["y"]
        y1 = y0 + height
        z0 = d["z"]
        z1 = z0 + depth
        datatype = getattr(np, d["datatype"])
        dataset_name = d["dataset_name"]
        dataset_id = d["dataset_id"]
        
        rh_logger.logger.report_event(
            "Writing %s: %d:%d, %d:%d, %d:%d" %
            (dataset_name, x0, x1, y0, y1, z0, z1))
        
        for subvolume, tif_path in d["blocks"]:
            svolume = Volume(**subvolume)
            sx0 = svolume.x
            sx1 = svolume.x1
            sy0 = svolume.y
            sy1 = svolume.y1
            sz0 = svolume.z
            sz1 = svolume.z1
            tif_dir = os.path.dirname(tif_path)
            if not os.path.isdir(tif_dir):
                os.makedirs(tif_dir)
            with tifffile.TiffWriter(tif_path, bigtiff=True) as fd:
                metadata = dict(x0=sx0, x1=sx1, y0=sy0, y1=sy1, z0=sz0, z1=sz1,
                                dataset_name = dataset_name,
                                dataset_id = dataset_id)
                block = data[sz0 - z0: sz1 - z0,
                             sy0 - y0: sy1 - y0,
                             sx0 - x0: sx1 - x0].astype(datatype)
                fd.save(block, 
                        photometric='minisblack',
                        compress=self.compression,
                        description=dataset_name,
                        metadata=metadata)
        rh_logger.logger.report_metric("Dataset store time (sec)",
                                       time.time() - t0)
        with self.open("w") as fd:
            json.dump(d, fd)
    
    def imread(self):
        '''Read the volume after it's written
        
        This is needed by neuroproof so that the Python can read the
        Neuroproof output.
        '''
        t0 = time.time()
        with open(self.storage_plan_path, "r") as fd:
            d = json.load(fd)
        depth, height, width = d["dimensions"]
        x0 = d["x"]
        x1 = x0 + width
        y0 = d["y"]
        y1 = y0 + height
        z0 = d["z"]
        z1 = z0 + depth
        datatype = getattr(np, d["datatype"])
        dataset_name = d["dataset_name"]
        dataset_id = d["dataset_id"]
        
        rh_logger.logger.report_event(
            "Reading %s: %d:%d, %d:%d, %d:%d" %
            (dataset_name, x0, x1, y0, y1, z0, z1))
        data = np.zeros((z1-z0, y1-y0, x1-x0), datatype)
        
        for subvolume, tif_path in d["blocks"]:
            svolume = Volume(**subvolume)
            sx0 = svolume.x
            sx1 = svolume.x1
            sy0 = svolume.y
            sy1 = svolume.y1
            sz0 = svolume.z
            sz1 = svolume.z1
            tif_dir = os.path.dirname(tif_path)
            block = tifffile.imread(tif_path)
            data[sz0 - z0: sz1 - z0,
                 sy0 - y0: sy1 - y0,
                 sx0 - x0: sx1 - x0] = block
        rh_logger.logger.report_metric("Dataset load time (sec)",
                                       time.time() - t0)
        return data
        

def write_loading_plan(loading_plan_path, volume_db, loading_plan_id):
    '''Write a loading plan to a file
    
    :param loading_plan_path: where to write the loading plan
    :param volume_db: the database containing the loading plan
    :param loading_plan_id: the ID of the loading plan record
    '''
    assert isinstance(volume_db, VolumeDB)
    
    volume = volume_db.get_loading_plan_volume(loading_plan_id)
    blocks = []
    for location, subvolume in \
        volume_db.get_subvolume_locations_by_loading_plan_id(loading_plan_id):
        blocks.append((location, subvolume.to_dictionary()))
    done_files = volume_db.get_dataset_paths_by_loading_plan_id(
        loading_plan_id)
    
    d = dict(
        dimensions=[volume.depth, volume.height, volume.width],
        x=volume.x,
        y=volume.y,
        z=volume.z,
        blocks=blocks,
        loading_plan_id=loading_plan_id,
        dataset_name=volume_db.get_loading_plan_dataset_name(loading_plan_id),
        datatype=volume_db.get_loading_plan_dataset_type(loading_plan_id),
        dataset_done_files=done_files
    )
    with open(loading_plan_path, "w") as fd:
        json.dump(d, fd)

def write_simple_loading_plan(
    loading_plan_path, image_filename, volume, dataset_name, dtype):
    '''Write a loading plan that reads from a single .tif file
    
    :param loading_plan_path: the path to the loading plan to be written
    :param image_filename: the filename of the .tif file
    :param volume: the extents and location of the volume to be written
    :param dataset_name: the name of the dataset, e.g. "image"
    :param dtype: the Numpy dtype, e.g. "uint8
    '''
    blocks = [(image_filename, volume.to_dictionary())]
    d = dict(
        dimensions=[volume.depth, volume.height, volume.width],
        x=volume.x,
        y=volume.y,
        z=volume.z,
        blocks=blocks,
        loading_plan_id=EMPTY_LOADING_PLAN_ID,
        dataset_name=dataset_name,
        datatype=dtype,
        dataset_done_files=[]
    )
    with open(loading_plan_path, "w") as fd:
        json.dump(d, fd)
        
class DestVolumeReader(object):
    '''A class for reading a volume using a loading plan
    
    '''
    def __init__(self, loading_plan_path):
        '''Initialize the target with the location of the loading plan

        :param loading_plan_path: the location of the loading plan on disk
        '''
        self.loading_plan_path = loading_plan_path
        self.__volume = None

    @property
    def volume(self):
        if self.__volume is None:
            with open(self.loading_plan_path, "r") as fd:
                d = json.load(fd)
                depth, height, width = d["dimensions"]
                self.__volume = Volume(d["x"], d["y"], d["z"],
                                       width, height, depth)
        return self.__volume
    
    @property
    def dataset_name(self):
        with open(self.loading_plan_path, "r") as fd:
            d = json.load(fd)
            return d["dataset_name"]
    
    def get_source_targets(self):
        '''Return the SrcVolumeTargets required to read this volume'''
        with open(self.loading_plan_path, "r") as fd:
            d = json.load(fd)
        tgts = []
        for path in d["dataset_done_files"]:
            tgts.append(SrcVolumeTarget(os.path.splitext(path)[0] + ".plan"))
        return tgts
    
    def imread(self):
        '''Read the volume'''
        t0 = time.time()
        with open(self.loading_plan_path, "r") as fd:
            d = json.load(fd)
        x0 = self.volume.x
        x1 = self.volume.x1
        y0 = self.volume.y
        y1 = self.volume.y1
        z0 = self.volume.z
        z1 = self.volume.z1
        dataset_name = d["dataset_name"]
        datatype = getattr(np, d["datatype"])
        rh_logger.logger.report_event(
            "Loading %s: %d:%d, %d:%d, %d:%d" % (
                dataset_name,
                x0, x1, y0, y1, z0, z1))

        result = None
        for tif_path, subvolume in d["blocks"]:
            svolume = Volume(**subvolume)
            if svolume.x >= x1 or\
               svolume.y >= y1 or\
               svolume.z >= z1 or \
               svolume.x1 <= x0 or \
               svolume.y1 <= y0 or \
               svolume.z1 <= z0:
                rh_logger.logger.report_event(
                    "Ignoring block %d:%d, %d:%d, %d:%d from load plan" %
                (svolume.x, svolume.x1, svolume.y, svolume.y1,
                 svolume.z, svolume.z1))
                continue
                
            with tifffile.TiffFile(tif_path) as fd:
                block = fd.asarray()
                if svolume.x == x0 and \
                   svolume.x1 == x1 and \
                   svolume.y == y0 and \
                   svolume.y1 == y1 and \
                   svolume.z == z0 and \
                   svolume.z1 == z1:
                    # Cheap win, return the block
                    rh_logger.logger.report_metric("Dataset load time (sec)",
                                                       time.time() - t0)
                    return block.astype(datatype)
                if result is None:
                    result = np.zeros((z1-z0, y1-y0, x1-x0), datatype)
                #
                # Defensively trim the block to within x0:x1, y0:y1, z0:z1
                #
                if svolume.z < z0:
                    block = block[z0-svolume.z:]
                    sz0 = z0
                else:
                    sz0 = svolume.z
                if svolume.z1 > z1:
                    block = block[:z1 - sz0]
                    sz1 = z1
                else:
                    sz1 = svolume.z1
                if svolume.y < y0:
                    block = block[:, y0-svolume.y:]
                    sy0 = y0
                else:
                    sy0 = svolume.y
                if svolume.y1 > y1:
                    block = block[:, :y1 - sy0]
                    sy1 = y1
                else:
                    sy1 = svolume.y1
                if svolume.x < x0:
                    block = block[:, :, x0-svolume.x:]
                    sx0 = x0
                else:
                    sx0 = svolume.x
                if svolume.x1 > x1:
                    block = block[:, :, :x1 - sx0]
                    sx1 = x1
                else:
                    sx1 = svolume.x1
                result[sz0 - z0:sz1 - z0,
                       sy0 - y0:sy1 - y0,
                       sx0 - x0:sx1 - x0] = block
        rh_logger.logger.report_metric("Dataset load time (sec)",
                                       time.time() - t0)
        return result

all = [SrcVolumeTarget, DestVolumeReader]