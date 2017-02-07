# Pickle A Classifier Help

This page documents the pickle-a-classifier GUI application. The application
creates a pickle file to be used by the Ariadne / Microns pipeline's classifier.
The pipeline expects the classifier to output several channels. For
segmentation:

* *membrane*: a single channel giving the membrane probabilities

or

* *x*, *y* and *z*: the channels for affinity map segmentation

For synapse predictions

* *synapse*: a single channel giving synapse probabilities

or
* *transmitter* and *receptor* for the presynaptic and postsynaptic parts
of a synapse

You may have two separate classifiers, one for segmentation and one for
synapse. These are combined using an *aggregate* classifier. The individual
classifiers may be *keras* or *caffe*. So the workflow is to run 
pickle-a-classifier once for the segmentation, once for the synapse and once
for the aggregate to combine the synapse and segmentation pickle files.

## Pickling a Keras model

Your classifier should have an input shape (the size of the image block
going into your classifier) and an output shape (the size of the image block
coming out of your classifier). In addition, you can crop your output, for
instance if you are unsure of how much border to trim.

Images are preprocessed with a normalization method. The normalization methods
are:
* *MATCH*: match each image plane's intensities against a reference distribution
and adjust the intensities so that there are the same proportion of voxels in
each intensity bin as in the reference distribution.
* *EQUALIZE_ADAPTHIST*: estimate the intensity in patches across the image
and equalize the image using an interpolation of the estimated intensity
in each patch.
* *RESCALE*: Clip the outlier intensities in each image plane and stretch
the remainder to the interval 0-255.
* *NONE*: pass the original un-normalized images into the classifier.

To pickle a Keras model.

* Select "keras" from the initial drop-down and press "Go".
* Hit the "Select" button for the model file and pick your Keras model .json
file.
* Hit the "Select" button for the weights file and pick your Keras model's
HDF5 weights file.
* Enter the input size. 
* Enter the output size.
* Optionally, enter the number of pixels to crop from either side of the
output block.
* For each class (in the order they are output), enter a class name in the
"Class names" field. Class names are separated by commas.
* Pick a normalization method
* Hit the "Save" button and enter a file name. Your classifier will be pickled
to a .pkl file.

## Pickling an aggregate classifier

To combine two classifiers, pickle them together with the aggregate classifier.
Each classifier has its associated pickle file. The aggregate classifier
saves the name of the pickle file along with the mapping of input class names
(as they were saved by you when you pickled the classifier) to output class
names. By default, the input names are passed through to the output.

To start, pick "aggregate" from the initial list and hit the "Go" button.

For each classifier:
* Press the "Add" button
* Pick a classifier file using the file dialog. The class channels produced
by the classifier will appear below the file name.
* For each class, optionally change the class name

When you are done, press "Save" and enter your pickle file's name. Your
aggregate classifier will be saved to the file you choose. If you make a 
mistake, you can start over by pressing "Go" again.

## Pickling a Caffe classifier

(currently not supported)
