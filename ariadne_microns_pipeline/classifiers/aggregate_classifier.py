'''A "classifier" that aggregates multiple classifiers

This classifier is a shim that aggregates the channels of multiple sub-
classifiers to create something that looks like a single classifier.

NB: mostly here for interim development - long-term goal is to have a
    single pixel classifier with multiple taps.
'''

from ..targets.classifier_target import AbstractPixelClassifier
import cPickle
import operator

class AggregateClassifier(AbstractPixelClassifier):
    '''A "classifier" that aggregates the channels from multiple sub-classifiers
    
    To use:
        Generate a pickle file for each of your classifiers.
        
        > from ariadne_microns_pipeline.classifiers.aggregate_classifier \
               import AggregateClassifier
        > import cPickle
        > my_classifier = AggregateClassifier(
             ["foo.pkl", "bar.pkl"],
             [dict(foo_in="foo_out"), dict(bar_in="bar_out")])
        > cPickle.dump(my_classifier, open("agg.pkl", "w"))
    '''
    
    def __init__(self, pickle_paths, name_maps):
        '''Initialize the classifier for pickling
        
        :param pickle_paths: sequence of paths to the pickle files of the
                             sub-classifiers.
        :param name_map: map of sub-classifier class names to aggregate
                         classifier class names. This is a sequence of
                         dictionaries in the same order as pickle_paths.
        '''
        self.pickle_paths = pickle_paths
        self.name_maps = name_maps
        self.__classifiers_loaded = False
    
    def __getstate__(self):
        return self.pickle_paths, self.name_maps
    
    def __setstate__(self, x):
        self.pickle_paths, self.name_maps = x
        self.__classifiers_loaded = False
        
    def classifiers(self):
        if not self.__classifiers_loaded:
            self.__classifiers = [
                cPickle.load(open(path, "r")) for path in self.pickle_paths]
            self.__classifiers_loaded = True
        return self.__classifiers
    
    def get_x_pad(self):
        return max(*map(lambda _:_.get_x_pad(), self.classifiers()))
    
    def get_y_pad(self):
        return max(*map(lambda _:_.get_y_pad(), self.classifiers()))
    
    def get_z_pad(self):
        return max(*map(lambda _:_.get_z_pad(), self.classifiers()))
    
    def get_class_names(self):
        return sum(operator.methodcaller("values"), self.name_maps, [])

    def get_resources(self, volume):
        d = {}
        for classifier in self.classifiers():
            resources = classifier.get_resources(volume)
            for key in resources:
                if key not in d:
                    d[key] = resources[key]
                elif d[key] < resources[key]:
                    d[key] = resources[key]
        return d

    def run_via_ipc(self):
        return any(map(operator.methodcaller("run_via_ipc"), 
                       self.classifiers()))
    
    def classify(self, image, x, y, z):
        d = {}
        x_pad = self.get_x_pad()
        y_pad = self.get_y_pad()
        z_pad = self.get_z_pad()
        for classifier, name_map in zip(self.classifiers(), self.name_maps):
            #
            # Get the cutout needed by the classifier
            #
            # Adjust based on difference between global and classifier
            # padding.
            #
            cx_pad = classifier.get_x_pad()
            cy_pad = classifier.get_y_pad()
            cz_pad = classifier.get_z_pad()
            x0 = x_pad - cx_pad
            x1 = image.shape[2] - x_pad + cx_pad
            y0 = y_pad - cy_pad
            y1 = image.shape[1] - y_pad + cy_pad
            z0 = z_pad - cz_pad
            z1 = image.shape[0] - z_pad + cz_pad
            dd = classifier.classify(image[z0:z1, y0:y1, x0:x1],
                                     x=x+x0, y=y+y0, z=z+z0)
            dd = dict([(name_map[k], v) for k, v in dd.items()])
            d.update(dd)
        return d
