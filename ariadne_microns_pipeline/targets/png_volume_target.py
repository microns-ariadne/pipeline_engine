import cv2
import luigi
import json
import numpy as np
import os

from utilities import shard

class PngVolumeTarget(luigi.LocalTarget):
    '''The PngVolumeTarget stores a volume as planes of .png files'''
    
    def __init__(self, paths, dataset_path, pattern, 
                 x, y, z, width, height, depth):
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
        super(PngVolumeTarget, self).__init__(self.__get_touchfile_name())
        
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
        super(PngVolumeTarget, self).__init__(self.__get_touchfile_name())
    
    def __get_filename(self, z):
        '''Get the file name for the plane at z'''
        return os.path.join(
            self.__get_dirname(),
            self.pattern.format(x=self.x, y=self.y, z=z) + ".png")
    
    def __get_dirname(self):
        return os.path.join(shard(self.paths, self.x, self.y, self.z),
                            self.dataset_path)
    
    def __get_touchfile_name(self):
        return self.__get_filename(self.z) + ".done"
    
    def imwrite(self, volume):
        '''Write the volume
        
        :param volume: a 3-d or 4d numpy array. If 4d, the last dimension
        is the color and must have a size of either 3 or 4. The dtype must
        be either uint8 or uint16. The coordinates are z, y, x and optionally
        color.
        '''
        self.makedirs()
        d = dict(dimensions=volume.shape,
                 dtype=volume.dtype.descr[0][1],
                 x=self.x,
                 y=self.y,
                 z=self.z,
                 filenames=[])
        for zidx in range(volume.shape[0]):
            filename = self.__get_filename(zidx + self.z)
            cv2.imwrite(filename, volume[zidx])
            d["filenames"].append(filename)

        with self.open(mode="w") as fd:
            json.dump(d, fd)
    
    def imread(self):
        with self.open(mode="r") as fd:
            d = json.load(fd)
        volume = np.zeros((self.depth, self.height, self.width), d["dtype"])
        for i, filename in enumerate(d["filenames"]):
            volume[i] = cv2.imread(filename, 2)
        return volume
    
    def create_volume(self, dtype, **kwargs):
        '''Create the volume in preparation for reading in parts.'''
        pass
    
    def imread_part(self, x, y, z, width, height, depth):
        '''Read a part of the volume
        
        :param x: the X offset of the subvolume to read in the global space
        :param y: the Y offset of the subvolume to read in the global space
        :param z: the Z offset of the subvolume to read in the global space
        :param width: the width of the subvolume to return
        :param height: the height of the subvolume to return
        :param depth; the depth of the subvolume to return
        '''
        #
        # TODO: optimize this
        #
        if not hasattr(self, "__volume"):
            self.__volume = self.imread()
        z0 = z - self.z
        z1 = z0 + depth
        y0 = y - self.y
        y1 = y0 + height
        x0 = x - self.x
        x1 = x0 + width
        return self.__volume[z0:z1, y0:y1, x0:x1]
    
    def imwrite_part(self, subvolume, x, y, z):
        '''Write a part of the volume
    
        Note that the subvolume's location is supplied in the global
        space, not the space of the target volume's array.
        
        Call "finish_volume()" when all parts are written
        
        :param subvolume: the subvolume to write to the target volume
        :param x: the x offset of the subvolume in the global space
        :param y: the y offset of the subvolume in the global space
        :param z: the z offset of the subvolume in the global space
        '''
        if not hasattr(self, "__volume"):
            shape = [self.depth, self.height, self.width]
            if subvolume.ndim == 4:
                shape.append(subvolume.shape[3])
            self.__volume = np.zeros(shape, subvolume.dtype)
            
        z0 = z - self.z
        z1 = z0 + subvolume.shape[0]
        y0 = y - self.y
        y1 = y0 + subvolume.shape[1]
        x0 = x - self.x
        x1 = x0 + subvolume.shape[2]
        self.__volume[z0:z1, y0:y1, x0:x1] = subvolume
    
    def finish_volume(self):
        self.imwrite(self.__volume)
        