import luigi
import json
import os

from ..parameters import Volume, DatasetLocation
from ..volumedb import VolumeDB

class SrcVolumeTarget(luigi.LocalTarget):
    '''A volume target that can be used to write a volume
    
    '''
    
    def __init__(self, volume_db_path, dataset_id):
        '''Initialize the target with the pathnames and file name pattern

        :param volume_db_path: the VolumeDB holding the knowledge of the write plan
        :param dataset_id: the ID of the dataset to write
        '''
        self.volume_db_path = volume_db_path
        self.dataset_id = dataset_id
        self.__volume = None
        super(VolumeTarget, self).__init__(self._get_touchfile_name())

    def __getstate__(self):
        return dict(paths=self.volume_db_path,
                    dataset_id = self.dataset_id)
    
    def __setstate__(self, state):
        self.volume_db_path = state["volume_db_path"]
        self.dataset_id = state["dataset_id"]
        super(VolumeTarget, self).__init__(self.__get_touchfile_name())
    
    def _get_touchfile_name(self):
        if self.__touchfile_name is not None:
            return self.__touchfile_name
        with VolumeDB(self.volume_db_path, "r") as db:
            return db.get_dataset_path(self.dataset_id)
    
    @property
    def volume(self):
        '''The global volume covered by this target'''
        if self.__volume is None:
            with VolumeDB(self.volume_db_path, "r") as db:
                self.__volume = db.get_dataset_volume(self.dataset_id)
        return self.__volume
    
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
        with VolumeDB(self.volume_db_path, "r") as db:
            db.imwrite(self.dataset_id, data)
            blocks = []
            for location, volume \
                in db.get_subvolume_locations_by_dataset_id(self.dataset_id):
                blocks.append([volume.to_dictionary(), location])
            d = dict(dimensions=[self.volume.depth,
                                 self.volume.height,
                                 self.volume.width],
                     x=self.volume.x,
                     y=self.volume.y,
                     z=self.volume.z,
                     blocks=blocks)
        with self.open("w") as fd:
            json.dump(blocks, fd)

class DestVolumeReader(object):
    '''A class for reading a volume using a loading plan
    
    '''
    def __init__(self, volume_db_path, loading_plan_id):
        '''Initialize the target with the pathnames and file name pattern

        :param volume_db_path: the VolumeDB holding the knowledge of the write plan
        :param loading_plan_id: the ID of the dataset to read
        '''
        self.volume_db_path = volume_db_path
        self.loading_plan_id = loading_plan_id
        self.__volume = None

    @property
    def volume(self):
        if __volume is None:
            with VolumeDB(self.volume_db_path, "r") as db:
                __volume = db.get_loading_plan_volume(self.loading_plan_id)
        return __volume
    
    def get_dataset_locations(self):
        '''Get the locations of the dataset .done files needed by the plan
        
        This can be used in the task's input() section to find the files
        that signal that the necessary datasets have been written.
        '''
        with VolumeDB(self.volume_db_path, "r") as db:
            return db.get_dataset_paths_by_loading_plan_id(self.loading_plan_id)
    
    def get_source_targets(self):
        '''Return the SrcVolumeTargets required to read this volume'''
        with VolumeDB(self.volume_db_path, "r") as db:
            return map(lambda dataset_id:SrcVolumeTarget(self.volume_db_path,
                                                         dataset_id),
                       db.get_loading_plan_dataset_ids(self.loading_plan_id))
    
    def imread(self):
        '''Read the volume'''
        with VolumeDB(self.volume_db_path, "r") as db:
            return db.imread(self.loading_plan_id)