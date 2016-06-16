#!/bin/bash

# 72-core
TOOLS_DIR=/home/armafire/tools
PIPELINE_ENGINE_DIR=/home/armafire/Pipeline/pipeline_engine
PIPELINE_ENGINE_EXEC_DIR=/home/armafire/Pipeline/pipeline_engine/exec_full

echo LD_PRELOAD=/home/armafire/Pipeline/tools/jemalloc/jemalloc-4.1.0/build/lib/libjemalloc.so
export LD_PRELOAD=/home/armafire/Pipeline/tools/jemalloc/jemalloc-4.1.0/build/lib/libjemalloc.so

echo export LD_LIBRARY_PATH=$TOOLS_DIR/cilkplus-install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TOOLS_DIR/cilkplus-install/lib64:$LD_LIBRARY_PATH

echo export LD_LIBRARY_PATH=$TOOLS_DIR/opencv-3-install-test/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TOOLS_DIR/opencv-3-install-test/lib:$LD_LIBRARY_PATH

echo export LD_LIBRARY_PATH=$TOOLS_DIR/vigra-install/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TOOLS_DIR/vigra-install/lib64:$LD_LIBRARY_PATH

echo export PYTHONPATH=$PIPELINE_ENGINE_EXEC_DIR:$PYTHONPATH
export PYTHONPATH=$PIPELINE_ENGINE_EXEC_DIR:$PYTHONPATH

echo export MATLABPATH=$PIPELINE_ENGINE_DIR/matlab_scripts
export MATLABPATH=export MATLABPATH=$PIPELINE_ENGINE_DIR/matlab_scripts

#/mnt/disk1/yaron/matlab_scripts
