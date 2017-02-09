#!/bin/bash
#
# classify.sh - example script to run the ClassifyTask
#
# Required environment variables:
#
# MICRONS_DIR - the location of the image downloaded by
#               download_from_butterfly.sh
# MICRONS_CLASSIFIER_PATH - the location of the classifier .pkl file
#
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
#  Use the same volume parameters as for download_from_butterfly.sh
#
#--------------------------------------------------

MICRONS_X=0
MICRONS_Y=0
MICRONS_Z=0
MICRONS_WIDTH=1496
MICRONS_HEIGHT=1496
MICRONS_DEPTH=50
#--------------------------------------------------
#
# These are the class names that the classifier produces.
# You can always tell the classifier to save only some of them.
#
# The script creates a .json dictionary mapping
# each class name to itself
#
#-------------------------------------------------

python -c 'import cPickle;import json; print json.dumps(dict([(_,_) for _ in cPickle.load(open("'$MICRONS_CLASSIFIER_PATH'")).get_class_names()]))' > $MICRONS_DIR/MICRONS_CLASS_NAMES
MICRONS_CLASS_NAMES=`cat $MICRONS_DIR/MICRONS_CLASS_NAMES`

#-------------------------------------------------
#
# Luigi needs the following resources to run this
#
# cpu_count=1 (one CPU to run the ClassifyTask)
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
cpu_count = 1
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
#-------------------------------------------------
#
# The Luigi command
#
#-------------------------------------------------
set -x
luigi  \
      --module=ariadne_microns_pipeline.tasks \
      ariadne_microns_pipeline.ClassifyTask \
      --classifier-path=$MICRONS_CLASSIFIER_PATH \
      --volume='{"x":'$MICRONS_X',"y":'$MICRONS_Y',"z":'$MICRONS_Z',"width":'$MICRONS_WIDTH',"height":'$MICRONS_HEIGHT',"depth":'$MICRONS_DEPTH'}' \
      --image-location='{"roots":["'$MICRONS_DIR'"],"dataset_name":"image","pattern":"{x:09d}_{y:09d}_{z:09d}_image"}' \
      --prob-roots='["'$MICRONS_DIR'"]' \
      --class-names="$MICRONS_CLASS_NAMES" \
      --pattern='{x:09d}_{y:09d}_{z:09d}'
set +x
#------------------------------------------------
#
# We need to be rude in order to exit. Kill the IPC tasks
#
#------------------------------------------------

if [ -z "$MICRONS_DONT_START_IPC" ]; then
    kill -9 $MICRONS_IPC_BROKER_PID
    kill -9 $MICRONS_IPC_WORKER_PID
fi
if [ -z "$MICRONS_DONT_RUN_LUIGID" ]; then
    kill -9 $MICRONS_LUIGID_PID
fi
