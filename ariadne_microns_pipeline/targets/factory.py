from ..volumedb import VolumeDB
from .volume_target import SrcVolumeTarget, DestVolumeReader

class TargetFactory(object):
    '''Produce volume targets
    
    To write a volume:
    * Get a dataset ID via get_dataset_id()
    * Create a task that uses it (e.g. as a parameter)
    * Call register_dataset to announce that your task will produce it
    * Call get_src_volume_target when you're ready to write it
    * Call imwrite() on the volume target to write it.
    
    To read a volume:
    * Get a loading plan ID via get_load_plan_id()
    * Create a task that uses the loading plan ID
    * Call register_dataset_dependent to announce that your task will use it
    * Call get_dest_volume_target when you're ready to read it
    * Call imread() on the volume to read it.
    
    '''
    def __init__(self, db_path):
        self.volume_db_path = db_path
        self.mode = mode
        
    def get_volume_target(self, dataset_id):
        '''Get a target that can store a volume
        
        :param dataset_id: the ID of the dataset previously registered
        with the database.
        '''
        return SrcVolumeTarget(self.volume_db_path, dataset_id)

    def get_volume_reader(self, loading_plan_id):
        '''Get a reader of the volume described by a loading plan
        
        :param loading_plan_id: the ID of the loading plan for loading
        a volume
        '''
        return DestVolumeReader(self.volume_db_path, loading_plan_id)