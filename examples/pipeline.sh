#!/bin/bash
#
# pipeline.sh
#
# This example demonstrates how to process a volume, creating a segmentation
# and the index.json and synapse-connection.json files.
#
# Environment variables
#
# MICRONS_CLASSIFIER_PATH - path to the .pkl file containing both the
#                           segmentation and synapse prediction classifiers.
#
# MICRONS_NP_CLASSIFIER_PATH - the path to your Neuroproof classifier
#
# MICRONS_SEGMENTATION_DIR - the global segmentation HDF5 file output by
#                             the pipeline gets stored in this directory
#
# MICRONS_SYNAPSE_FILE - contains the detected pre- and post-synaptic
#                        pairs along with the coordinates of the synapses.
#
# MICRONS_SEGMENTATION_STATISTICS_FILE - the path to the .csv file
#                        containing block statistics comparing the segmentation
#                        to ground-truth. A .pdf is also generated that displays
#                        the results visually.
#
# MICRONS_SYNAPSE_STATISTICS_FILE - the path to the .json file that compares the
#                        ground-truth synapse data to the detected.
#

#--------------------------------------------------
#
# Intermediate files are stored by default in /tmp/examples/pipeline,
# but you can define MICRONS_DIR to put them elsewhere
#
#--------------------------------------------------

if [ -z "$MICRONS_DIR" ]; then
    MICRONS_DIR=/tmp/examples/pipeline
fi
mkdir -p $MICRONS_DIR

#---------------------------------------------------
#
# Where is Butterfly? You can use a local Butterfly by
# redefining BUTTERFLY_API_URL
#
#---------------------------------------------------
if [ -z "$BUTTERFLY_API_URL" ]; then
   BUTTERFLY_API_URL=https://butterfly.rc.fas.harvard.edu/api
fi

#----------------------------------------------------
#
# Start ipc workers
#
# The broker's output is saved to ipc-broker.{log,err}
# The worker's output is saved to ipc-worker.{log,err}
#
# If you have the broker and worker(s) started, define
# MICRONS_DONT_START_IPC
#----------------------------------------------------

if [ -z "$MICRONS_DONT_START_IPC" ]; then
    microns-ipc-broker >> $MICRONS_DIR/ipc-broker.log \
                       2>> $MICRONS_DIR/ipc-broker.err.log &
    MICRONS_IPC_BROKER_PID="$!"
    microns-ipc-worker \
        >> $MICRONS_DIR/ipc-worker.log \
        2>> $MICRONS_DIR/ipc-worker.err.log &
    MICRONS_IPC_WORKER_PID="$!"
fi

#---------------------------------------------------
#
#  This is the inner volume of the ECS dataset
#
# There's some tricky math here to get the padding
# out of your classifier.
#
#--------------------------------------------------

MICRONS_X_PAD=`python -c 'import cPickle;print cPickle.load(open("'$MICRONS_CLASSIFIER_PATH'")).get_x_pad()'`
MICRONS_Y_PAD=`python -c 'import cPickle;print cPickle.load(open("'$MICRONS_CLASSIFIER_PATH'")).get_y_pad()'`
MICRONS_Z_PAD=`python -c 'import cPickle;print cPickle.load(open("'$MICRONS_CLASSIFIER_PATH'")).get_z_pad()'`

MICRONS_X=$MICRONS_X_PAD
MICRONS_Y=$MICRONS_Y_PAD
MICRONS_Z=$MICRONS_Z_PAD
MICRONS_WIDTH=$((1496-$MICRONS_X_PAD-$MICRONS_X_PAD))
MICRONS_HEIGHT=$((1496-$MICRONS_Y_PAD-$MICRONS_Y_PAD))
MICRONS_DEPTH=$((97-$MICRONS_Z_PAD-$MICRONS_Z_PAD))

#-------------------------------------------------
#
# Luigi needs the following resources to run this
#
# cpu_count=2 (two CPUs to run Neuroproof)
# gpu_count=1 (one GPU to run the ClassifyTask)
# memory=??? (enough memory to run the task. Memory is measured in bytes.)
#
# We construct a luigi.cfg file that has this and
# point to it with the LUIGI_CONFIG_PATH environment variable
#-------------------------------------------------
export LUIGI_CONFIG_PATH=$MICRONS_DIR/luigi.cfg
cat <<EOF > "$LUIGI_CONFIG_PATH"
[core]
no_configure_logging=True
[resources]
cpu_count = 2
gpu_count = 1
memory=30000000000
EOF
#-------------------------------------------------
#
# Run the Luigi daemon
#
#-------------------------------------------------

if [ -z "$MICRONS_DONT_RUN_LUIGID" ]; then
    luigid --logdir=$MICRONS_DIR \
           >> $MICRONS_DIR/luigid.log \
           2>> $MICRONS_DIR/luigid.err &
    MICRONS_LUIGID_PID="$!"
fi

#--------------------------------------------------
#
# Run the Luigi task
#
#--------------------------------------------------
set -x

luigi --module ariadne_microns_pipeline.pipelines \
      ariadne_microns_pipeline.PipelineTask \
      --experiment=ECS_iarpa_201610_gt_4x6x6 \
      --sample=neocortex \
      --dataset=sem \
      --synapse-channel=synapses \
      --url=$BUTTERFLY_API_URL \
      --pixel-classifier-path=$MICRONS_CLASSIFIER_PATH \
      --neuroproof-classifier-path=$MICRONS_NP_CLASSIFIER_PATH \
      --volume='{"x":'$MICRONS_X',"y":'$MICRONS_Y',"z":'$MICRONS_Z',"width":'$MICRONS_WIDTH',"height":'$MICRONS_HEIGHT',"depth":'$MICRONS_DEPTH'}' \
      --temp-dirs='["'$MICRONS_DIR'"]' \
      --block-width=$MICRONS_WIDTH \
      --block-height=$MICRONS_HEIGHT \
      --wants-transmitter-receptor-synapse-maps \
      --statistics-csv-path=$MICRONS_SEGMENTATION_STATISTICS_FILE \
      --synapse-statistics-path=$MICRONS_SYNAPSE_STATISTICS_FILE \
      --stitched-segmentation-location=$MICRONS_SEGMENTATION_DIR \
      --synapse-connection-location=$MICRONS_SYNAPSE_FILE
set +x
#---------------------------------------------------
#
# kill the subprocesses
#
#---------------------------------------------------

if [ -z "$MICRONS_DONT_START_IPC" ]; then
    kill -9 $MICRONS_IPC_BROKER_PID
    kill -9 $MICRONS_IPC_WORKER_PID
fi
if [ -z "$MICRONS_DONT_RUN_LUIGID" ]; then
    kill -9 $MICRONS_LUIGID_PID
fi
