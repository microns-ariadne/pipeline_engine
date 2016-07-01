import enum
from hdf5_target import HDF5VolumeTarget
from png_volume_target import PngVolumeTarget

class TFEnums(enum.Enum):
    use_png_volume=1
    use_hdf5_volume=2
    
class TargetFactory(object):
    
    def __init__(self):
        self.volume_target_type = TFEnums.use_png_volume
        
    def get_volume_target(self, paths, dataset_path, pattern, x, y, z, 
                          width, height, depth,
                          target_type=None):
        '''Get a target that can store a volume
        
        :param paths: a sequence of root paths, each of which is on a different
             disk spindle.
        :dataset_path: for individual files, a subdirectory to contain the
             files. For HDF5, the dataset_path is the name of the dataset
             within the HDF5 file.
        :param pattern: A pattern for str.format(). The variables available
        are "x", "y" and "z". Example: "{x:04}_{y:04}_{z:04}.png" yields
        "0001_0002_0003.png" for a plane with X offset 1, Y offset 2 and
        Z offset 3.
        The touchfile (indicating that all PNG files are available) is
        "pattern.format(x=x, y=y, z=z)+".done".
        :param x: the X offset of the volume in the global space
        :param y: the Y offset of the volume in the global space
        :param z: the Z offset of the volume in the global space
        :param width: the width of the volume
        :param height: the height of the volume
        :param depth: the depth of the volume
        '''
        if target_type is None:
            target_type = self.volume_target_type
        if target_type == TFEnums.use_hdf5_volume:
            return HDF5VolumeTarget(
                paths, dataset_path, pattern, x, y, z, width, height, depth)
        else:
            return PngVolumeTarget(
                paths, dataset_path, pattern, x, y, z, width, height, depth)