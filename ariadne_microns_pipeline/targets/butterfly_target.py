'''Butterfly Luigi target'''

from cv2 import imdecode
import json
import luigi
import numpy as np
from tornado.httpclient import HTTPClient, HTTPResponse
from urllib2 import HTTPError

class ButterflyTarget(luigi.Target):
    '''The ButterflyTarget gets a Butterfly cutout plane from the server'''
    
    def __init__(self,
                 experiment,
                 sample,
                 dataset,
                 channel,
                 x, y, z, width, height, url="http://localhost:2001/api"):
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
            "&height=%d" % self.height
        client = HTTPClient()
        response = client.fetch(url)
        assert isinstance(response, HTTPResponse)
        if response.code >= 400:
            raise HTTPError(
                url, response.code, response.reason, response.headers, None)
        body = np.frombuffer(response.body, np.uint8)
        return imdecode(body, 0)


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
    
    def __cache_channel_params(self):
        '''Make the REST channel_metadata call to get the volume extents'''
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

def get_butterfly_plane_from_channel(channel_target, x, y, z, width, height):
    '''Get a butterfly plane from a butterfly channel target
    
    :param channel_target: A Butterfly channel target
    :param x: the x offset of the plane
    :param y: the y offset of the plane
    :param z: the plane # of the plane
    :param width: the width of the plane
    :param height: the height of the plane
    '''
    return ButterflyTarget(
        experiment=channel_target.experiment,
        sample=channel_target.sample,
        dataset=channel_target.dataset,
        channel=channel_target.channel,
        x=x, y=y, z=z, width=width, height=height,
        url=channel_target.url)

all=[get_butterfly_plane_from_channel, ButterflyChannelTarget, ButterflyTarget]