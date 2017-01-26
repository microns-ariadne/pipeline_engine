import cv2
import luigi
import json
import numpy as np
import os
import rh_logger
import time

from .utilities import shard
from .volume_target import VolumeTarget

class PngVolumeTarget(VolumeTarget):
    '''The PngVolumeTarget stores a volume as planes of .png files'''

    def __new__(cls, *args, **kwargs):
        self = VolumeTarget.__new__(cls, *args, **kwargs)
        self.__has_volume = False
        return self
        
    def __get_filename(self, z):
        '''Get the file name for the plane at z'''
        return os.path.join(
            self.__get_dirname(),
            self.pattern.format(x=self.x, y=self.y, z=z) + ".png")
    
    def __get_dirname(self):
        return os.path.join(shard(self.paths, self.x, self.y, self.z),
                            self.dataset_path)
    
    def imwrite(self, volume):
        '''Write the volume
        
        :param volume: a 3-d or 4d numpy array. If 4d, the last dimension
        is the color and must have a size of either 3 or 4. The dtype must
        be either uint8 or uint16. The coordinates are z, y, x and optionally
        color.
        '''
        t0 = time.time()
        for path in self.paths:
            tgt_dir = os.path.join(path, self.dataset_path)
            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
        for zidx in range(volume.shape[0]):
            filename = self.__get_filename(zidx + self.z)
            img = volume[zidx]
            if volume.dtype.itemsize == 4:
                # Do uint32 as 3 colors
                img = np.dstack((
                    img & 0xff,
                    (img / 256) & 0xff,
                    (img / 65536) & 0xff)).astype(np.uint8)
            cv2.imwrite(filename, img)

        self.finish_imwrite(volume.dtype)
        rh_logger.logger.report_metric("PngVolumeTarget.imwrite (sec)", 
                                       time.time() - t0)

    def anticipate_filenames(self):
        '''Return the filenames that we anticipate writing'''
        return [self.__get_filename(z) 
                for z in range(self.z, self.z + self.depth)]
    
    def finish_imwrite(self, dtype):
        '''Fake writing a volume
        
        It's assumed that something else called anticipate_filenames to find
        out how to write files, then it did it. We finish by writing the
        json output file.
        '''
        d = dict(dimensions=[self.volume.depth, 
                             self.volume.height, 
                             self.volume.width],
                 dtype=dtype.descr[0][1],
                 x=self.x,
                 y=self.y,
                 z=self.z,
                 filenames=self.anticipate_filenames())
        with self.open(mode="w") as fd:
            json.dump(d, fd)
    
    def get_filenames(self):
        '''Get the filenames for the individual planes of the volume
        
        Reads the filenames from the .json file written by imwrite.
        '''
        with self.open(mode="r") as fd:
            d = json.load(fd)
            return d["filenames"]
        
    def imread(self):
        t0 = time.time()
        with self.open(mode="r") as fd:
            d = json.load(fd)
        volume = np.zeros((self.depth, self.height, self.width), d["dtype"])
        for i, filename in enumerate(d["filenames"]):
            img = cv2.imread(
                filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) \
                .astype(volume.dtype)
            if img.ndim == 3:
                img = img[:, :, 0] +\
                    img[:, :, 1] * 256 +\
                    img[:, :, 2] * 65536
            volume[i] = img
        rh_logger.logger.report_metric("PngVolumeTarget.imread (sec)", 
                                       time.time() - t0)
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
        if not self.__has_volume:
            self.__volume = self.imread()
            self.__has_volume = True
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
        if not self.__has_volume:
            shape = [self.depth, self.height, self.width]
            if subvolume.ndim == 4:
                shape.append(subvolume.shape[3])
            self.__volume = np.zeros(shape, subvolume.dtype)
            self.__has_volume = True
            
        z0 = z - self.z
        z1 = z0 + subvolume.shape[0]
        y0 = y - self.y
        y1 = y0 + subvolume.shape[1]
        x0 = x - self.x
        x1 = x0 + subvolume.shape[2]
        self.__volume[z0:z1, y0:y1, x0:x1] = subvolume
    
    def finish_volume(self):
        self.imwrite(self.__volume)
    
    @staticmethod
    def from_done_file(path, pattern):
        '''Create a PngVolumeTarget from its .done file
        
        :param path: path to .done file
        :param pattern: the pattern for naming files, e.g. 
        {x:09d}_{y:09d}_{z:09d}
        '''
        d = json.load(open(path, "r"))
        roots = list(set([os.path.dirname(os.path.dirname(filename))
                          for filename in d["filenames"]]))
        dataset_name = os.path.split(os.path.dirname(d["filenames"][0]))[1]
        depth, height, width = d["dimensions"]
        x = d["x"]
        y = d["y"]
        z = d["z"]
        result = PngVolumeTarget(roots, dataset_name, pattern, x, y, z,
                                 width, height, depth, touchfile_name=path)
        return result
        