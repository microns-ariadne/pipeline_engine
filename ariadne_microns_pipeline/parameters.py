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
    
    def __repr__(self):
        return "Volume: x=%d:%d, y=%d:%d, z=%d:%d" % (
            self.x, self.x1, self.y, self.y1, self.z, self.z1)
    

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
    
    def __repr__(self):
        return "DatasetLocation: [%s].%s (%s)" % (
            ",".join(self.roots), self.dataset_name, self.pattern)

'''A dataset location for a dataset that doesn't exist

If a dataset is optional, use this to signify that the user doesn't
want it.
'''
EMPTY_DATASET_LOCATION = DatasetLocation([], "None", "")

def is_empty_dataset_location(location):
    '''Check to see if a dataset location is the empty dataset'''
    return len(location.roots) == 0

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
        d = dict(roots=tuple(x.roots), 
                 dataset_name=x.dataset_name, 
                 pattern=x.pattern)
        return json.dumps(d)

class MultiVolumeParameter(luigi.Parameter):
    '''A parameter representing a number of volumes taken together
    
    This is useful when merging or considering adjacent or overlapping volumes.
    The MultiVolumeParameter represents a number of volumes that act as
    inputs for some merging operation. The serialization format is a list
    of dictionaries with keys, "volume" and "location" giving the Volume
    and DatasetLocation of some input volume.
    '''
    
    def parse(self, x):
        l = json.loads(x)
        result = []
        for d in l:
            dv = d["volume"]
            dl = d["location"]
            result.append(dict(
                volume=Volume(x=dv["x"],
                              y=dv["y"],
                              z=dv["z"],
                              width=dv["width"],
                              height=dv["height"],
                              depth=dv["depth"]),
                location=DatasetLocation(roots=dl["roots"], 
                                         dataset_name=dl["dataset_name"], 
                                         pattern=dl["pattern"])))
        return result
    
    def serialize(self, x):
        l = []
        for d in x:
            volume = d["volume"]
            location = d["location"]
            l.append(dict(volume=dict(x=volume.x,
                                      y=volume.y,
                                      z=volume.z,
                                      width=volume.width,
                                      height=volume.height,
                                      depth=volume.depth),
                          location=dict(roots=location.roots,
                                        dataset_name=location.dataset_name,
                                        pattern=location.pattern)))
        return json.dumps(l)

class MultiDatasetLocationParameter(luigi.Parameter):
    '''A parameter representing an indeterminate number of datasets
    
    This is useful when aggregating a number of datasets that are defined
    over the same volume.
    '''
    def parse(self, x):
        l = json.loads(x)
        result = []
        for d in l:
            result.append(DatasetLocation(roots=d["roots"],
                                          dataset_name=d["dataset_name"],
                                          pattern=d["pattern"]))
    
    def serialize(self, x):
        result = []
        for dataset_location in x:
            result.append(dict(roots=dataset_location.roots,
                               dataset_name=dataset_location.dataset_name,
                               pattern=dataset_location.pattern))
        return json.dumps(result)
    
all = [Volume, VolumeParameter, DatasetLocation, DatasetLocationParameter,
       MultiVolumeParameter]