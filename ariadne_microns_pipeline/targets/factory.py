import enum
from hdf5_target import HDF5VolumeTarget
from png_volume_target import PngVolumeTarget

class TFEnums(enum.Enum):
    use_png_volume=1
    use_hdf5_volume=2
    
class TargetFactory(object):
    
    def __init__(self):
        self.volume_target_type = TFEnums.use_png_volume
        
    def get_volume_target(self, location, volume,
                          target_type=None):
        '''Get a target that can store a volume
        
        :param location: A DatasetLocation giving the paths, dataset_name
        and file pattern for naming files
        :param volume: the offset and extents of the volume
        :param target_type: either TFEnums.use_png_volume to have one .png
            file per plane, TFEnums.use_hdf5_volume to use a single HDF5
            dataset for the volume.
        '''
        if target_type is None:
            target_type = self.volume_target_type
        if target_type == TFEnums.use_hdf5_volume:
            return HDF5VolumeTarget(
                location.roots, location.dataset_name, location.pattern,
                volume.x, volume.y, volume.z, 
                volume.width, volume.height, volume.depth)
        else:
            return PngVolumeTarget(
                location.roots, location.dataset_name, location.pattern,
                volume.x, volume.y, volume.z, 
                volume.width, volume.height, volume.depth)
