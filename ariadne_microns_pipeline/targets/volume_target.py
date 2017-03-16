import luigi
import os

from .utilities import shard
from ..parameters import Volume, DatasetLocation


class VolumeTarget(luigi.LocalTarget):
    '''An abstract volume target
    
    This class has the framework for maintaining the volume and supplying
    some useful helpers. Derived classes should implement `imread()` and
    `imwrite()`.
    '''
    
    def __init__(self, paths, dataset_path, pattern, 
                 x, y, z, width, height, depth, touchfile_name=None):
        '''Initialize the target with the pathnames and file name pattern

        :param paths: A list of paths. For a plane, Z, we write the png file
        to the Zth path modulo len(paths),.
        :param dataset_path: in this case, the subdirectory for the .PNG files
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
        :param touchfile_name: the name of the touchfile (.done file). Default
                               is constructed as described above.
        '''
        self.paths = paths
        self.dataset_path = dataset_path
        self.pattern = pattern
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth
        self.__has_volume = False
        self.__touchfile_name = touchfile_name
        super(VolumeTarget, self).__init__(self._get_touchfile_name())

    def __getstate__(self):
        return dict(paths=self.paths,
                    dataset_path=self.dataset_path,
                    pattern=self.pattern,
                    x=self.x,
                    y=self.y,
                    z=self.z,
                    width=self.width,
                    height=self.height,
                    depth=self.depth)
    
    def __setstate__(self, state):
        self.paths = state["paths"]
        self.dataset_path = state["dataset_path"]
        self.pattern = state["pattern"]
        self.x = state["x"]
        self.y = state["y"]
        self.z = state["z"]
        self.width = state["width"]
        self.height = state["height"]
        self.depth = state["depth"]
        self.__touchfile_name = None
        super(VolumeTarget, self).__init__(self._get_touchfile_name())
    
    def _get_touchfile_name(self):
        if self.__touchfile_name is not None:
            return self.__touchfile_name
        root = shard(self.paths, self.x, self.y, self.z)
        filename = "{x:09d}_{y:09d}_{z:09d}_{dataset_path:s}.done".format(
            **self.__getstate__())
        return os.path.join(root, filename)
    
    @property
    def volume(self):
        '''The global volume covered by this target'''
        return Volume(self.x, self.y, self.z, 
                      self.width, self.height, self.depth)
    
    @property
    def dataset_location(self):
        '''The location on disk of this target's data'''
        return DatasetLocation(self.paths, self.dataset_path, self.pattern)