#!/bin/bash

# 8-core
# PIPELINE_BASE=/home/amatveev/Pipeline
# TOOLS_BASE=/home/amatveev/Pipeline/tools
# 
# echo export LD_LIBRARY_PATH=$PIPELINE_BASE/cilkplus-install/lib64:$TOOLS_BASE/OpenCV/opencv-2.4-install/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$PIPELINE_BASE/cilkplus-install/lib64:$TOOLS_BASE/OpenCV/opencv-2.4-install/lib:$LD_LIBRARY_PATH
# 
# echo export LD_PRELOAD=$TOOLS_BASE/jemalloc/jemalloc-3.0.0/lib/libjemalloc.so
# export LD_PRELOAD=$TOOLS_BASE/jemalloc/jemalloc-3.0.0/lib/libjemalloc.so

# 72-core CM2
TOOLS_BASE=/home/armafire/tools

echo export LD_LIBRARY_PATH=$TOOLS_BASE/cilkplus-install/lib64:$TOOLS_BASE/opencv-3-install-test/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TOOLS_BASE/cilkplus-install/lib64:$TOOLS_BASE/opencv-3-install-test/lib:$LD_LIBRARY_PATH
 
#echo export LD_PRELOAD=/home/armafire/Pipeline/tools/jemalloc/jemalloc-4.1.0/build/lib/libjemalloc.so
#export LD_PRELOAD=/home/armafire/Pipeline/tools/jemalloc/jemalloc-4.1.0/build/lib/libjemalloc.so

echo export LD_PRELOAD=/home/armafire/Pipeline/tools/jemalloc/jemalloc-4.1.0/build/lib/libjemalloc.so
export LD_PRELOAD=/home/armafire/Pipeline/tools/jemalloc/jemalloc-4.1.0/build/lib/libjemalloc.so

