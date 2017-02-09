# Ariadne / Microns pipeline examples

This directory has a number of examples of Luigi commands for the Ariadne /
Microns pipeline. This assumes you have access to the butterfly server at http://butterfly.rc.fas.harvard.edu:2001.

## download_from_butterfly.sh

This example runs Luigi with a local scheduler to download the training volume to the location specified by the environment variable $MICRONS_DIR. An example invocation:

    MICRONS_DIR=/tmp/microns-data download_from_butterfly.sh

## classify.sh

This example classifies the volume downloaded from buttterfly by the
script above. You should provide a pickled classifier. You can pickle a
classifier using the `pickle-a-classifier` GUI app or you can use a typical
classifier such as
`/n/coxfs01/leek/classifiers/2016-12-08/membrane_2016-12-08.pkl`. Keras and
other GPU applications need to reserve the GPU and Keras needs some time
in order to compile the classifier, which it does every time the process
starts. Because of this, we use the `microns-ipc-broker` and
`microns-ipc-worker` in the script to maintain a process that persists for
the entire Luigi command (which, in the case of a pipeline, can be hundreds
of invocations of the ClassifyTask).

The script takes two environment parameters:

* *MICRONS_DIR*: the directory in which you stored the image volume
* *MICRONS_CLASSIFIER_PATH*: the path to your pickled classifier

An example invocation:

    MICRONS_DIR=/tmp/examples \
    MICRONS_CLASSIFIER_PATH=/n/coxfs01/leek/classifiers/2016-12-08/membrane_2016-12-08.pkl \
    ./classify.sh

## np_train.sh

This example trains Neuroproof on the ECS_train_images dataset using the
pipeline. The example runs a number of sub-tasks to download the datasets,
run the classifier, perform the oversegmentation and train Neuroproof,
resulting in a Neuroproof classifier file.

You'll have to configure your .rh-config.yaml file to point to the pipeline's
version of Neuroproof (see the header for np_train.sh for directions) and
you may have to find or compile Neuroproof itself with this project's
makefile.

The script takes two environment parameters:

* *MICRONS_CLASSIFIER_PATH*: this is the path to your .pkl classifier file
* *MICRONS_NP_CLASSIFIER_PATH*: the pipeline saves your classifier here. Please
use a ".xml" extension.

The pipeline has quite a number of tuning parameters and these are set to
their defaults. You might want to use this example as a jumping-off point
for your own training script instead of assuming that it will give you a
good classifier.