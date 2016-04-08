#!/bin/bash

# 72-core
TOOLS_BASE=/home/armafire/tools

echo export LD_LIBRARY_PATH=$TOOLS_BASE/cilkplus-install/lib64:$TOOLS_BASE/opencv-install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TOOLS_BASE/cilkplus-install/lib64:$TOOLS_BASE/opencv-install/lib:$LD_LIBRARY_PATH

echo export LD_PRELOAD=$TOOLS_BASE/jemalloc/jemalloc-3.0.0/lib/libjemalloc.so
export LD_PRELOAD=$TOOLS_BASE/jemalloc/jemalloc-3.0.0/lib/libjemalloc.so

echo export LD_LIBRARY_PATH=/home/armafire/tools/vigra-install/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/armafire/tools/vigra-install/lib64:$LD_LIBRARY_PATH

echo export MATLABPATH=/home/armafire/Pipeline/matlab_scripts/:/mnt/disk1/yaron/matlab_scripts
export MATLABPATH=/home/armafire/Pipeline/matlab_scripts/:/mnt/disk1/yaron/matlab_scripts

