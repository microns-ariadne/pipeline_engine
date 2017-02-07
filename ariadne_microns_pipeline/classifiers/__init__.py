from aggregate_classifier import AggregateClassifier
from caffe_classifier import CaffeClassifier
from keras_classifier import KerasClassifier

all_classifiers = dict(aggregate=AggregateClassifier,
                       caffe=CaffeClassifier,
                       keras=KerasClassifier)
