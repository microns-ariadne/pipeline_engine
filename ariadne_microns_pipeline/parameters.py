'''Ariadne/Microns specific Luigi parameters'''

import luigi
import json

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
    
    def to_dictionary(self):
        '''Get the dictionary representation of the volume
        
        Note: given a volume, v, you can reconstruct it by doing
        Volume(**v.to_dictionary())
        '''
        return dict(x=self.x, y=self.y, z=self.z, 
                    width=self.width, height=self.height, depth=self.depth)
    
    def contains(self, x, y, z):
        '''Return True if the volume contains the given point
        
        :param x: the x coordinate of the point
        :param y: the y coordinate of the point
        :param z: the z coordinate of the point
        '''
        return x >= self.x and x < self.x1 and\
               y >= self.y and y < self.y1 and\
               z >= self.z and z < self.z1
    
    def overlaps(self, other):
        '''Return True if the volume overlaps another volume
        
        :param other: other volume
        '''
        return self.x < other.x1 and other.x < self.x1 and\
               self.y < other.y1 and other.y < self.y1 and\
               self.z < other.z1 and other.z < self.z1
    
    def __str__(self):
        return "Volume: x=%d:%d, y=%d:%d, z=%d:%d" % (
            self.x, self.x1, self.y, self.y1, self.z, self.z1)
    
    def __repr__(self):
        return "Volume(x=%d, y=%d, z=%d, width=%d, height=%d, depth=%d)" % \
               (self.x, self.y, self.z, self.width, self.height, self.depth)
    
    def __eq__(self, other):
        if isinstance(other, Volume):
            return self.x == other.x and self.y == other.y and \
                   self.z == other.z and self.width == other.width  and \
                   self.height == other.height and self.depth == other.depth
        return False

class VolumeParameter(luigi.Parameter):
    '''Describes a volume.
    
    The format is a JSON dictionary with keys of x, y, z, width, height and
    depth.
    '''
    
    def parse(self, x):
        '''Parse an individual value'''
        
        d = json.loads(x)
        return Volume(x=d["x"],
                      y=d["y"],
                      z=d["z"],
                      width=d["width"],
                      height=d["height"],
                      depth=d["depth"])
    
    def serialize(self, x):
        d = dict(x=x.x, 
                 y=x.y,
                 z=x.z,
                 width=x.width,
                 height=x.height,
                 depth=x.depth)
        return json.dumps(d)    

'''A dataset ID for a dataset that doesn't exist

If a dataset is optional, use this to signify that the user doesn't
want it.
'''
EMPTY_DATASET_ID = 0

EMPTY_LOADING_PLAN_ID = 0

def is_empty_dataset_id(dataset_id):
    '''Check to see if a dataset location is the empty dataset'''
    return dataset_id == 0

'''A default location on disk

e.g. one that's synthesized from a root + a suffix.
'''
DEFAULT_LOCATION = ":DEFAULT_LOCATION:"

'''A location indicating an unused element, e.g. don't produce this file'''
EMPTY_LOCATION = "/dev/null"

all = [Volume, VolumeParameter, EMPTY_DATASET_ID, is_empty_dataset_id,
       DEFAULT_LOCATION, EMPTY_LOCATION]