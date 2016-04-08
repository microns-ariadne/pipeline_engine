#!/bin/bash

#####################################################################
# Params
#####################################################################
IS_FORCE=false

RUN_NAME=$1

LABELS_DIR=$2

WS_PROBS_DIR=$3
NP_PROBS_DIR=$3

NP_CLASSIFIER_DIR=/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-AC3-train-53-w2-QTD-64-GT/
#/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-K11-AC-256-3D-PAD/
#/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-K11-AC-256-3D-PAD/
#/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-P3-errors-3D-49x49-32f-GT/
#/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-P3-errors-NEW-2-3D-49x49-32f-GT-ws-2D
#/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-P3-errors-3D-49x49-32f-GT/
#/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-P3-errors-NEW-2-3D-49x49-32f-GT-ws-sv-4/

#####################################################################
# Env setup
#####################################################################

source ./neuroproof_agg/init_env_neuroproof.sh

#####################################################################
# Exec
#####################################################################

echo "========================================================================="
echo "RUNNING PIPELINE"
echo "========================================================================="
echo "Parameters:"
echo "-------------------------------------------------------------------------"
echo "  force             = $IS_FORCE"
echo "  run-name          = $RUN_NAME"
echo "  labels-dir        = $LABELS_DIR"
echo "  ws-probs-dir      = $WS_PROBS_DIR"
echo "  np-probs-dir      = $NP_PROBS_DIR"
echo "  np-classifier-dir = $NP_CLASSIFIER_DIR"
echo "========================================================================="

PARAMS=""

if [ "$IS_FORCE" == true ]; then
    PARAMS+="--force  " 
fi

PARAMS+=" \
    --run-name=$RUN_NAME \
    --labels-dir=$LABELS_DIR \
    --ws-probs-dir=$WS_PROBS_DIR \
    --np-probs-dir=$NP_PROBS_DIR \
    --np-classifier-dir=$NP_CLASSIFIER_DIR"
          
echo "python execute_pipeline.py $PARAMS"

python execute_pipeline.py $PARAMS
