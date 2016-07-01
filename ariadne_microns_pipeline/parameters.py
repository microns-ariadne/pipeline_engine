'''Ariadne/Microns specific Luigi parameters'''

import luigi

class Volume(object):
    '''A 3-d volume'''
    
    def __init__(self, x, y, z, width, height, depth):
        """Initialize the volume
        
        :param x: The left edge of the volume in global coordinates
        :param y: The top edge of the volume
        :param z: the first plane of the volume
        :param width: the width of the volume
        :param height: the height of the volume
        :param depth: the number of planes in the volume
        """
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth
        
    @property
    def x1(self):
        """The left edge of the volume, exclusive"""
        return self.x + self.width
    @property
    def y1(self):
        """The bottom edge of the volume, exclusive"""
        return self.y + self.height
    
    @property
    def z1(self):
        """One past the last plane of the volume"""
        return self.z + self.depth
    
class VolumeParameter(luigi.Parameter):
    '''Describes a volume.
    
    The format is "x=#,y=#,z=#,width=#,height=#,depth=#".
    '''
    