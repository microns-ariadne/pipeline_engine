'''A reference classifier

This classifier inverts the image and rank-orders the pixels to come up
with probabilities.
'''
from ..targets.classifier_target import AbstractPixelClassifier
import numpy as np

class ReferenceClassifier(AbstractPixelClassifier):
    
    def get_x_pad(self):
        return 0
    
    def get_y_pad(self):
        return 0
    
    def get_z_pad(self):
        return 0
    
    def get_class_names(self):
        return ["membrane"]
    
    def classify(self, image, x, y, z):
        stack = np.zeros(image.shape, np.uint8)
        for z in range(image.shape[0]):
            bins = np.bincount(image[z].flatten())
            cum_bins = np.cumsum(bins)
            total = np.prod(image.shape[1:])
            prob = 511 - 511 * cum_bins.astype(float) / total
            prob[prob < 256] = 256
            prob = (prob - 256).astype(np.uint8)
            stack[z] = prob[image[z]]
        return dict(membrane=stack)