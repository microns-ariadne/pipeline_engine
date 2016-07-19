'''"Classify" by reading predictions from Butterfly

To use:

    > import cPickle
    > from ariadne_microns_pipeline.classifiers.butterfly_classifier\
          import ButterflyClassifier
    > cPickle.dump(ButterflyClassifier(experiment=<butterfly-experiment-name>,
                                       sample=<butterfly-sample-name>,
                                       dataset=<butterfly-dataset-name>,
                                       channel=<butterfly-channel-name>,
                                       url=<butterfly-url>),
                    open(<picke-file>, "w"))

where
<pickle-file> is the name that you'll supply as the --classifier-path
    in the ClassifiyTask.
<butterfly-url> is the URL of the butterfly server.

On Butterfly, you'll need to add a channel in your .rh-config.yaml file
with the classification data.
'''

import numpy as np

from ..targets.classifier_target import AbstractPixelClassifier
from ..targets.butterfly_target import ButterflyTarget


class ButterflyClassifier(AbstractPixelClassifier):
    '''Read classifications from butterfly'''
    
    def __init__(self,
                 experiment,
                 sample,
                 dataset,
                 channel,
                 url="http://localhost:2001/api"):
        '''
        
        :param experiment: The parent experiment
        :param sample: The identifier of the biological sample that was imaged
        :param dataset: The identifier of a conceptual volume that was imaged
        :param channel: The name of a channel defined within the conceptual
            volume
        :param url: the URL of the butterfly server, e.g. http://localhost:2001
        '''
        self.experiment = experiment
        self.sample = sample
        self.dataset = dataset
        self.channel = channel
        self.url = url
    
    def __getstate__(self):
        return dict(experiment=self.experiment,
                    sample=self.sample,
                    dataset=self.dataset,
                    channel=self.channel,
                    url=self.url)
    
    def __setstate__(self, x):
        self.experiment = x["experiment"]
        self.sample = x["sample"]
        self.dataset = x["dataset"]
        self.channel = x["channel"]
        self.url = x["url"]
    
    def get_class_names(self):
        return ["membrane"]
    
    def get_x_pad(self):
        return 0
    
    def get_y_pad(self):
        return 0
    
    def get_z_pad(self):
        return 0
    
    def classify(self, image, x, y, z):
        volume = np.zeros(image.shape, np.uint8)
        for zi in range(z, z+image.shape[0]):
            btarget = ButterflyTarget(
                experiment=self.experiment,
                sample=self.sample,
                dataset=self.dataset,
                channel=self.channel,
                x=x, y=y, z=zi,
                width=image.shape[2],
                height=image.shape[1],
                url=self.url)
            volume[zi-z]=btarget.imread()
        return dict(membrane=volume)