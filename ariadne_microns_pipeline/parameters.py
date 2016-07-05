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


class DatasetLocation(object):
    '''A dataset location (see DatasetLocationParameter)'''
    def __init__(self, roots, dataset_name, pattern):
        self.roots = roots
        self.dataset_name = dataset_name
        self.pattern = pattern


class DatasetLocationParameter(luigi.Parameter):
    '''The particulars necessary for describing the location of a volume
    
    A volume on disk can be sharded across multiple drives. Its name
    has three parts: a root (which is the sharded path), a dataset
    (which is a subdirectory of the root for planes and is the dataset
    name for HDF5 datasets) and a pattern which is used for naming files.
    
    The pattern for volumes uses str.format(x=x, y=y, z=z) for formatting.
    A typical pattern might be "{x:09d}_{y:09d}_{z:09d}"
    
    The format is a JSON dictionary with keys of "roots", "dataset_name" and
    "pattern"
    '''
    
    def parse(self, x):
        d = json.loads(x)
        return DatasetLocation(d["roots"], d["dataset_name"], d["pattern"])
    
    def serialize(self, x):
        d = dict(roots=x.roots, 
                 dataset_name=x.dataset_name, 
                 pattern=x.pattern)
        return json.dumps(d)