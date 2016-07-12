'''HDF5 Luigi targets

:HDF5VolumeTarget - a representation of a 3-D volume
:HDF5DatasetTarget - a raw dataset [TO-DO]
'''

import luigi
import h5py
import os
from utilities import shard

class HDF5VolumeTarget(luigi.File):
    '''An HDF5 dataset target representing a 3-D volume
    
    The 3-d volume is traditionally arranged as Z, Y, X so that Z-strides
    are the largest. This makes reading a plane a reasonably efficient operation.
    
    There are two ways of setting the volume. You can call `imwrite()` with
    the entire volume and you're done or you can call `create_volume()`, then
    call `imwrite_part` repeatedly, then call `finish_volume()` when you are
    done.
    '''
    
    def __init__(self, paths, dataset_path, pattern, x, y, z,
                 width, height, depth):
        '''Initialize the target
        
        :param path: path to the HDF5 file
        :param dataset_path: path to the dataset within the HDF5 file
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
        super(HDF5VolumeTarget, self).__init__(self.__get_fullpath())
        self.h5path = self.__get_hdf5_path()
    
    def __get_hdf5_path(self):
        '''The path to the HDF5 file'''
        return os.path.join(
            shard(self.paths, self.x, self.y, self.z),
            self.pattern.format(x=self.x, y=self.y, z=self.z) + ".h5")
    
    def __get_fullpath(self):
        '''The path to the marker file for the dataset
        
        You can't reliably detect when an HDF dataset is written because
        the write is not atomic. This will change in HDF5 1.10
        (https://www.hdfgroup.org/HDF5/docNewFeatures/)
        
        So we maintain a separate file per dataset for this purpose.
        '''
        return self.__get_hdf5_path() + "." + self.dataset_path + ".done"
    
    def imread(self):
        '''Read the volume
        
        Returns the entire volume. Axes are Z, Y, X.
        '''
        with h5py.File(self.h5path, "r") as fd:
            return fd[self.dataset_path][:]
    
    def imread_part(self, x0, x1, y0, y1, z0, z1):
        '''Read a portion of the volume
        
        :param x0: the X offset in the global space of the left side of the 
                   block
        :param x1: the X offset in the global space of the right side of the
                   block (non-inclusive)
        :param y0: the y offset in the global space of the top of the block
        :param y1: the y offset in the global space of the bottom of the block
                   (non-inclusive)
        :param z0: the z offset in the global space of the first plane in
                   the block
        :param z1: the z offset in the global space one past the last plane
                   in the block.
        '''
        x0, x1 = [_ - self.x for _ in x0, x1]
        y0, y1 = [_ - self.y for _ in x0, x1]
        z0, z1 = [_ - self.z for _ in z0, z1]
        with h5py.File(self.h5path, "r") as fd:
            return fd[self.dataset_path][z0:z1, y0:y1, x0:x1]
    
    def imwrite(self, volume, **kwargs):
        '''Write the volume
        
        :param volume: the volume which should have axes of Z, Y and X
        :param **kwargs: Keyword arguments passed into 
            h5py.Group.create_dataset. 
            See http://docs.h5py.org/en/latest/high/dataset.html for some
            hints about compressing a dataset or changing its block structure.
        '''
        pathdir = os.path.dirname(self.path)
        if not os.path.isdir(pathdir):
            os.makedirs(pathdir)
        with h5py.File(self.h5path, "a") as fd:
            if self.dataset_path in fd:
                del fd[self.dataset_path]
            ds = fd.create_dataset(
                self.dataset_path, data=volume, **kwargs)
            ds.attrs["x"] = self.x
            ds.attrs["y"] = self.y
            ds.attrs["z"] = self.z
            ds.attrs["target_class"] = "volume"
        self.finish_volume()
        
    def finish_volume(self):
        '''Mark the volume as completely written'''
        open(self.path, "w").write("done")

    def create_volume(self, dtype, **kwargs):
        '''Create an empty volume
        
        :param dtype: the data type, e.g. numpy.uint8 or bool
        :param **kwargs: keyword arguments passed into create_dataset
        such as the block structure or compression. 
        (see http://docs.h5py.org/en/latest/high/dataset.html)
        '''
        pathdir = os.path.dirname(self.path)
        if not os.path.isdir(pathdir):
            os.makedirs(pathdir)
        with h5py.File(self.h5path, "a") as fd:
            if self.dataset_path in fd:
                del fd[self.dataset_path]
            assert isinstance(fd, h5py.Group)
            ds = fd.create_dataset(self.dataset_path,
                                   shape=(self.depth, self.height, self.width),
                                   dtype=dtype, **kwargs)
            ds.attrs["x"] = self.x
            ds.attrs["y"] = self.y
            ds.attrs["z"] = self.z
            ds.attrs["target_class"] = "volume"

    def imwrite_part(self, subvolume, x, y, z):
        '''Write a part of the volume
        
        Note that the subvolume's location is supplied in the global
        space, not the space of the target volume's array.
        
        :param subvolume: the subvolume to write to the target volume
        :param x: the x offset of the subvolume in the global space
        :param y: the y offset of the subvolume in the global space
        :param z: the z offset of the subvolume in the global space
        '''
        x0 = x - self.x
        x1 = x0 + subvolume.shape[2]
        y0 = y - self.y
        y1 = y0 + subvolume.shape[1]
        z0 = z - self.z
        z1 = z0 + subvolume.shape[0]
        with h5py.File(self.h5path, "a") as fd:
            ds = fd[self.dataset_path]
            ds[z0:z1, y0:y1, x0:x1] = subvolume


class HDF5FileTarget(luigi.File):
    '''An HDF5 file encompassing multiple targets
    
    If a task produces multiple volumes, it's often easiest to put them
    all in the same HDF5 file. The individual volume targets can then
    be accessed from the same file, but the production of all of them is
    atomic because a single target is produced.
    '''
    
    def __init__(self, path, dataset_paths):
        '''Initialize the file target
        
        :param path: the path to the HDF5 file
        :param dataset_paths: the dataset names of each dataset to be produced
                              within the HDF5 file.
        '''
        super(HDF5FileTarget, self).__init__(path)
        self.dataset_paths = dataset_paths
    
    def get_subtarget(self, name):
        '''Get the named subtarget
        
        :param name: the dataset name
        '''
        with h5py.File(self.path, "r") as fd:
            ds = fd[name]
            target_class = ds.attrs["target_class"]
            if target_class == "volume":
                return HDF5VolumeTarget(self.path, name)
            raise ValueError("Unknown target class: %s" % target_class)
    
    def exists(self):
        '''Return True if all subtargets exist'''
        for dataset_path in dataset_paths:
            if not self.get_subtarget(dataset_path).exists():
                return False
        return True