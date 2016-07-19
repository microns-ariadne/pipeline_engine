import luigi
import cPickle
import os

from abc import ABCMeta
from luigi.six import add_metaclass

@add_metaclass(ABCMeta)
class AbstractPixelClassifier:
    '''The pickled classifier should implement this interface'''
    
    def get_x_pad(self):
        '''Pad images by this much in the X direction
        
        The returned image will have this much removed from it from the
        left and right. The input image should be padded by this much on
        either side.
        '''
        raise NotImplementedError()
    
    def get_y_pad(self):
        '''Pad images by this much in the Y direction
        
        The returned image will have this much removed from it from the
        top and bottom. The input image should be padded by this much on
        either side.
        '''
        raise NotImplementedError()
    
    def get_z_pad(self):
        '''Pad images by this much in the Z direction
        
        The returned image will have this many planes removed from it
        from above and below.
        '''
        raise NotImplementedError()
    
    def get_class_names(self):
        '''Return the names of the classes produced by the classifier'''
        raise NotImplementedError()
    
    def classify(self, image, x, y, z):
        '''Classify the image
        
        :param image: an image that is appropriate for the classifier, padded
            by the amounts given by get_x_pad... etc
        :param x: the offset of the image volume in the X direction
        :param y: the offset of the image volume in the Y direction
        :param z: the offset of the image volume in the Z direction
        :returns: a dictionary of probability outputs with the keys being
        the class names.
        '''
        raise NotImplementedError()


class PixelClassifierTarget(luigi.File):
    '''A pixel classifier target. You can use this target to classify pixels'''
    
    # A cache of classifier pickle pathname to two-tuple of
    #    classifier and file modification time
    #
    cache = {}
    
    def __init__(self, path):
        super(PixelClassifierTarget, self).__init__(path)
        self.__classifier = None
        
    def __getstate__(self):
        return self.path
    
    def __setstate__(self, path):
        self.path = path
        self.__classifier = None
        
    @property
    def classifier(self):
        if self.__classifier is None:
            mtime = os.stat(self.path).st_mtime
            if self.path in self.cache:
                classifier, old_mtime = self.cache[self.path]
                if mtime == old_mtime:
                    return classifier
                
            unpickler = cPickle.Unpickler(self.open())
            self.__classifier = unpickler.load()
            self.cache[self.path] = (self.__classifier, mtime)
        return self.__classifier
    
    #
    # Convenience methods
    #
    @property
    def x_pad(self):
        '''The amount of padding to add to the left and right side of the image

        '''
        return self.classifier.get_x_pad()
    
    @property
    def y_pad(self):
        '''The amount of padding to add to the top and bottom of the image

        '''
        return self.classifier.get_y_pad()
    
    @property
    def z_pad(self):
        '''The amount of padding to add before and after the image

        '''
        return self.classifier.get_z_pad()
    
    def classify(self, image):
        '''Classify the image
        
        :param image: an image that is appropriate for the classifier, padded
            by the amounts given by get_x_pad... etc
        :returns: a dictionary of probability outputs with the keys being
        the class names.
        '''
        return self.classifier.classify(image)