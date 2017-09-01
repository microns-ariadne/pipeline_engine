'''Butterfly Luigi target'''

import json
import luigi
import numpy as np
import tifffile
from tornado.httpclient import HTTPClient, HTTPResponse
from cStringIO import StringIO
from urllib2 import HTTPError

#
# This stupid monkey patch fixes an incompatibility between numpy 1.13 and
# tifffile.
#
if tuple(map(int, np.version.version.split("."))) >= (1, 13):
    old_fromfile = np.fromfile
    def fromfile(*args, **kwargs):
        try:
            return old_fromfile(*args, **kwargs)
        except TypeError:
            raise IOError("Work-around for tifffile")
    np.fromfile = fromfile

class ButterflyTarget(luigi.Target):
    '''The ButterflyTarget gets a Butterfly cutout plane from the server'''
    
    def __init__(self,
                 experiment,
                 sample,
                 dataset,
                 channel,
                 x, y, z, width, height,
                 url="http://localhost:2001/api",
                 resolution=0):
        '''Initialize the butterfly target
        
        :param experiment: The parent experiment of the plane of data
        :param sample: The identifier of the biological sample that was imaged
        :param dataset: The identifier of a conceptual volume that was imaged
        :param channel: The name of a channel defined within the conceptual
            volume, for instance "raw" for the original data or "seg" for
            the segmentation annotations.
        :param x: The X offset of the cutout within the dataset volume.
        :param y: The Y offset of the cutout within the dataset volume.
        :param z: The Z plane of the cutout within the dataset volume.
        :param width: the width of the cutout
        :param height: the height of the cutout
        :param url: The REST endpoint for getting a chunk of data
        :param resolution: MIPMAP level. A resolution of zero is 1:1,
                           a resolution of one is 2:1, a resolution of
                           two is 4:1, etc. Resolution is only in the x/y
                           directions.
        
        All coordinates and dimensions are in the downsampled space of
        the resolution. For instance x=1000, y=1000, z=200, width=1024,
        height=1024, depth=10, resolution=1 is the downsampled volume of
        x=2000, y=2000, z=200, width=2048, height=2048, depth=10, resolution=0.
        '''
        self.experiment = experiment
        self.sample = sample
        self.dataset = dataset
        self.channel= channel
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.url = url
        self.resolution = resolution
        self.channel_target = ButterflyChannelTarget(
            experiment, sample, dataset, channel, url)
    
    def exists(self):
        '''Does the target exist?
        
        The ButterflyTarget always returns True, assuming that it would be
        an operational error for the Butterfly server not to be able
        to serve the plane.
        '''
        return True
    
    def imread(self):
        '''Read the volume, returning a Numpy array'''
    
        url = "%s/data" % self.url +\
            "?experiment=%s" % self.experiment +\
            "&sample=%s" % self.sample +\
            "&dataset=%s" % self.dataset +\
            "&channel=%s" % self.channel +\
            "&x=%d" % self.x +\
            "&y=%d" % self.y +\
            "&z=%d" % self.z +\
            "&width=%d" % self.width +\
            "&height=%d" % self.height +\
            "&format=tif"+\
            "&view=grayscale"
        if self.resolution != 0:
            url += "&resolution=%d" % self.resolution
        client = HTTPClient()
        response = client.fetch(url)
        assert isinstance(response, HTTPResponse)
        if response.code >= 400:
            raise HTTPError(
                url, response.code, response.reason, response.headers, None)
        body = StringIO(response.body)
        result = tifffile.imread(body).astype(self.channel_target.data_type)
        return result

class ButterflyChannelTarget(luigi.Target):
    '''Represents a channel on the volume of a Butterfly dataset'''
    
    def __init__(self, experiment, sample, dataset, channel, url):
        '''Initialize the channel
        
        :param experiment: The parent experiment of the plane of data
        :param sample: The identifier of the biological sample that was imaged
        :param dataset: The identifier of a conceptual volume that was imaged
        :param channel: The name of a channel defined within the conceptual
            volume, for instance "raw" for the original data or "seg" for
            the segmentation annotations.
        :param url: The REST endpoint for the Butterfly API
        '''
        self.experiment = experiment
        self.sample = sample
        self.dataset = dataset
        self.channel = channel
        self.url = url
        self.__fetched = False
    
    def exists(self):
        try:
            self.__cache_channel_params()
            return True
        except:
            return False
    
    @property
    def x_extent(self):
        self.__cache_channel_params()
        return self.__x_extent
    
    @property
    def y_extent(self):
        self.__cache_channel_params()
        return self.__y_extent
    
    @property
    def z_extent(self):
        self.__cache_channel_params()
        return self.__z_extent
    
    @property
    def data_type(self):
        self.__cache_channel_params()
        return getattr(np, self.__data_type)
    
    def __cache_channel_params(self):
        '''Make the REST channel_metadata call to get the volume extents'''
        if self.__fetched:
            return
        url = "%s/channel_metadata" % self.url +\
            "?experiment=%s" % self.experiment +\
            "&sample=%s" % self.sample +\
            "&dataset=%s" % self.dataset +\
            "&channel=%s" % self.channel
        client = HTTPClient()
        response = client.fetch(url)
        assert isinstance(response, HTTPResponse)
        if response.code >= 400:
            raise HTTPError(
                url, response.code, response.reason, response.headers, None)
        d = json.loads(response.body)
        self.__x_extent = d["dimensions"]["x"]
        self.__y_extent = d["dimensions"]["y"]
        self.__z_extent = d["dimensions"]["z"]
        self.__data_type = d["data-type"]
        self.__fetched = True

def get_butterfly_plane_from_channel(
    channel_target, x, y, z, width, height, resolution=0):
    '''Get a butterfly plane from a butterfly channel target
    
    :param channel_target: A Butterfly channel target
    :param x: the x offset of the plane
    :param y: the y offset of the plane
    :param z: the plane # of the plane
    :param width: the width of the plane
    :param height: the height of the plane
    :param resolution: the MIPMAP resolution at which to retrieve.
    '''
    return ButterflyTarget(
        experiment=channel_target.experiment,
        sample=channel_target.sample,
        dataset=channel_target.dataset,
        channel=channel_target.channel,
        x=x, y=y, z=z, width=width, height=height,
        url=channel_target.url,
        resolution=resolution)

class LocalButterflyChannelTarget(luigi.Target):
    '''Represents a channel on the volume of a local Butterfly dataset'''
    
    def __init__(self, index_file):
        '''Initialize the channel
        
        :param index_file: the index file that has the butterfly tiles
        
        See LocalButterflyTask for a description of this file
        '''
        self.index_file = index_file
        self.__fetched = False
    
    def exists(self):
        try:
            self.__cache_channel_params()
            return True
        except:
            return False
    
    @property
    def x_extent(self):
        self.__cache_channel_params()
        return self.__x_extent
    
    @property
    def y_extent(self):
        self.__cache_channel_params()
        return self.__y_extent
    
    @property
    def z_extent(self):
        self.__cache_channel_params()
        return self.__z_extent
    
    @property
    def data_type(self):
        self.__cache_channel_params()
        return getattr(np, self.__data_type)
    
    def __cache_channel_params(self):
        '''extract info from the index file'''
        if self.__fetched:
            return
        index = json.load(open(self.index_file))
        self.__z_extent = len(index["sections"])
        dimensions = index["dimensions"]
        self.__x_extent = dimensions["n_columns"] * dimensions["width"]
        self.__y_extent = dimensions["n_rows"] * dimensions["height"]
        self.__fetched = True

all=[get_butterfly_plane_from_channel, ButterflyChannelTarget, ButterflyTarget,
     LocalButterflyChannelTarget]