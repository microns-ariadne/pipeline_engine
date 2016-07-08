'''A multivolume target contains multiple channels for a given volume

The use case is a classifier which outputs multiple probability maps in one
go - in that case, you want a task that has a single multivolume target as
its output
'''

import luigi
import os
from ..parameters import Volume, DatasetLocation

class MultivolumeTarget(luigi.LocalTarget):
    '''A target with multiple channels for a given volume'''
    
    def __init__(self, paths, channels, pattern,
                 x, y, z, width, height, depth):
        '''Initialize the target with paths and a volume
        
        :param paths: A list of paths. We shard the volume among the paths.
        :param channels: The names of the channels of the datasets
        in the multivolume. These are used to select a dataset when using
        imread and to write a dataset using imwrite.
        :param pattern: A pattern for str.format(). The variables available
        are "x", "y" and "z". The channel name is tacked onto the end, separated
        from the body with a dash.
        Example: "{x:04d}_{y:04d}_{z:04d}" yields
        "0001_0002_0003-membrane.png" for a plane with X offset 1, Y offset 2
        and Z offset 3 for the "membrane" channel.
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
        self.channels = channels
        self.pattern = pattern
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth
        super(MultivolumeTarget, self).__init__(self.__get_touchfile_name())
        
    def __getstate__(self):
        return dict(paths=self.paths,
                    channels=self.channels,
                    pattern=self.pattern,
                    x=self.x,
                    y=self.y,
                    z=self.z,
                    width=self.width,
                    height=self.height,
                    depth=self.depth)
    
    def __setstate__(self, state):
        self.paths = state["paths"]
        self.channels = state["channels"]
        self.pattern = state["pattern"]
        self.x = state["x"]
        self.y = state["y"]
        self.z = state["z"]
        self.width = state["width"]
        self.height = state["height"]
        self.depth = state["depth"]
        super(MultivolumeTarget, self).__init__(self.__get_touchfile_name())
    
    def __get_touchfile_name(self):
        return os.path.join(
            self.paths[0], 
            self.pattern.format(x=self.x, y=self.y, z=self.z) + "-" + 
            "-".join(sorted(self.channels)))
    
    def get_channel(self, channel):
        '''Return a channel's volume target
        
        :param channel: the name of the channel
        :returns: the volume target for that channel, e.g. as would be
        returned by calling `TargetFactory().get_volume_target()`
        '''
        from .factory import TargetFactory
        
        return TargetFactory().get_volume_target(
            location=DatasetLocation(
                roots=self.paths,
                dataset_name=channel,
                pattern=self.pattern + "-" + channel),
            volume=Volume(self.x, self.y, self.z,
                          self.width, self.height, self.depth))
    
    def imread(self, channel):
        '''Read the volume from a given channel'''
        return self.get_channel(channel).imread()
    
    def imwrite(self, channel, volume):
        '''Write a volume to a channel
        
        If all channels have been written, we write the "done" file as well.
        '''
        self.get_channel(channel).imwrite(volume)
        for channel in self.channels:
            if not self.get_channel(channel).exists():
                break
        else:
            with open(self.__get_touchfile_name(), "w") as fd:
                fd.write("done\n")