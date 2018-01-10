"""Use a classification stored in an HDF5 file

To use:

    > import cPickle
    > from ariadne_microns_pipeline.classifiers.hdf_classifier \
          import HDF5Classifier
    > cPickle.dump(HDF5Classifier(
        channel_names=["z", "y", "z"],
        blockdescs=[
        (<filename>, <xoff>, <yoff>, <zoff>, <width>, <height>, <depth>),
        ...
        (<filename>, <xoff>, <yoff>, <zoff>, <width>, <height>, <depth>)]),
        open(<picklefile>, "w"))
   
   You can provide multiple HDF5 files and stitch them together
"""

import h5py
import numpy as np
from ..parameters import Volume
from ..targets.classifier_target import AbstractPixelClassifier
from scipy.ndimage import zoom

class BlockDescriptor(object):
    def __init__(self, filename, xoff, yoff, zoff, width, height, depth):
        self.filename = filename
        self.volume = Volume(xoff, yoff, zoff, width, height, depth)

class HDF5Classifier(AbstractPixelClassifier):
    '''Read classifications from HDF5'''
    
    def __init__(self, channel_names, blockdescs, 
                 resolution=0, 
                 axes_indexes=(0, 1, 2, 3)):
        '''
        
        :param channel_names: the volume's channel names, e.g. 
        "z-affinity", "y-affinity", "x-affinity" in the order they appear in
        the HDF file.
        :param blockdescs: a descriptor for each HDF5 file consisting of
        a tuple of filename, x offset, y offset, z offset, width, height
        and depth.
        :param resolution: the mipmap level of the data within the hdf5 file
        :param axes_indexes: the index of the channel, z, y, and x in that order
        in the HDF5 file
        '''
        self.channel_names = tuple(channel_names)
        self.resolution = resolution
        self.axes_indexes=tuple(axes_indexes)
        self.blockdescs = []
        for blockdesc in blockdescs:
            self.blockdescs.append(BlockDescriptor(*blockdesc))
    
    def __getstate__(self):
        state = dict(
            channel_names=self.channel_names,
            resolution=self.resolution,
            axes_indexes = self.axes_indexes,
            blockdescs=[
                (_.filename, _.volume.x, _.volume.y, _.volume.z, 
                 _.volume.width, _.volume.height, _.volume.depth)
                for _ in self.blockdescs])
        return state
    
    def __setstate__(self, x):
        self.channel_names = x["channel_names"]
        self.resolution=x["resolution"]
        self.axes_indexes = x["axes_indexes"]
        self.blockdescs = [
            BlockDescriptor(filename, xoff, yoff, zoff, width, height, depth)
            for filename, xoff, yoff, zoff, width, height, depth 
            in x["blockdescs"]]
    
    def get_x_pad(self):
        return 0
    
    def get_y_pad(self):
        return 0
    
    def get_z_pad(self):
        return 0
    
    def classify(self, image, x, y, z):
        result = np.zeros((len(self.channel_names),
                           image.shape[0], image.shape[1], image.shape[2]), 
                          np.uint8)
        volume = Volume(x, y, z, image.shape[2], image.shape[1], image.shape[0])
        m = 2 ** self.resolution
        for blockdesc in self.blockdescs:
            if volume.overlaps(blockdesc.volume):
                v = volume.get_overlapping_region(blockdesc.volume)
                x0s = (v.x - blockdesc.volume.x) / m
                x1s = (v.x1 + m - 1 - blockdesc.volume.x) / m
                y0s = (v.y - blockdesc.volume.y) / m
                y1s = (v.y1 + m - 1- blockdesc.volume.y) / m
                z0s = (v.z - blockdesc.volume.z)
                z1s = (v.z1 - blockdesc.volume.z)
                x0d = (v.x - volume.x)
                x1d = (v.x1 - volume.x)
                y0d = (v.y - volume.y)
                y1d = (v.y1 - volume.y)
                z0d = (v.z - volume.z)
                z1d = (v.z1 - volume.z)
                slices = [None] * 4
                slices[self.axes_indexes[0]] = slice(0, len(self.channel_names))
                slices[self.axes_indexes[1]] = slice(z0s, z1s)
                slices[self.axes_indexes[2]] = slice(y0s, y1s)
                slices[self.axes_indexes[3]] = slice(x0s, x1s)
                slices = tuple(slices)
                with h5py.File(blockdesc.filename, "r") as fd:
                    block = \
                        fd[fd.keys()[0]][slices].transpose(self.axes_indexes)
                if m > 1:
                    block = zoom(block, (1, 1, m, m), order=1)
                offxs = (v.x - blockdesc.volume.x) % m
                offys = (v.y - blockdesc.volume.y) % m
                result[:, z0d:z1d, y0d:y1d, x0d:x1d] = \
                    block[:, 0:z1d-z0d, 
                          offys:y1d - y0d + offys,
                          offxs:x1d - x0d + offxs]
        return dict([(channel_name, result[i])
                     for i, channel_name in enumerate(self.channel_names)])
                    