#! /bin/bash
#
# np_train.sh
#
# This script trains Neuroproof using a classifier and the ECS_train_images
# dataset. You should have Neuroproof installed and should have the following
# in your .rh-config.yaml file:
#
# neuroproof:
#   neuroproof_graph_learn: <path-to-neuroproof-graph-learn>
#   ld_library_path:
#   - <path-to-boost-install-libs>
#   - <path-to-cilkplus-install-libs>
#   - <path-to-jsoncpp-install-libs>
#   - <path-to-opencv-install-libs>
#   - <path-to-vigra-install-libs>
#
# Intermediate files are written to /tmp/examples/nptrain.
#
# Environment variables
#
# MICRONS_CLASSIFIER_PATH - path to your classifier's .pkl file
# MICRONS_NP_CLASSIFIER_PATH - path for the neuroproof classifier .XML file
#
if [ -z "$MICRONS_DIR" ]; then
    MICRONS_DIR=/tmp/examples/nptrain
fi

#---------------------------------------------------
#
# Where is Butterfly?
#
#---------------------------------------------------
if [ -z "$BUTTERFLY_API_URL" ]; then
   BUTTERFLY_API_URL=http://butterfly.rc.fas.harvard.edu:2001/api
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
cpu_count = 4
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
      --experiment=ECS_train_images \
      --sample=neocortex \
      --dataset=sem \
      --url=$BUTTERFLY_API_URL \
      --pixel-classifier-path=$MICRONS_CLASSIFIER_PATH \
      --neuroproof-classifier-path=$MICRONS_NP_CLASSIFIER_PATH \
      --volume='{"x":'$MICRONS_X',"y":'$MICRONS_Y',"z":'$MICRONS_Z',"width":'$MICRONS_WIDTH',"height":'$MICRONS_HEIGHT',"depth":'$MICRONS_DEPTH'}' \
      --temp-dirs='["'$MICRONS_DIR'"]' \
      --block-width=$MICRONS_WIDTH \
      --block-height=$MICRONS_HEIGHT \
      --block-depth=$MICRONS_DEPTH \
      --wants-neuroproof-learn
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
