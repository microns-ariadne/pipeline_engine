#!/bin/bash

#####################################################################
# Run config
#####################################################################

IS_FORCE="--force"

# Z_RANGE=0-2
# X_RANGE=2-3
# Y_RANGE=2-3
 
# Z_RANGE=0-2
# X_RANGE=4-6
# Y_RANGE=4-6
 
Z_RANGE=0-2
X_RANGE=0-14
Y_RANGE=0-14

SECTION_RANGE=0-1849

#####################################################################
# Exec
#####################################################################

# ./run_pipeline.sh CNN_POSTPROCESS $Z_RANGE $X_RANGE $Y_RANGE $SECTION_RANGE $IS_FORCE 
# 
# ./run_pipeline.sh WS_NP_PREPARE $Z_RANGE $X_RANGE $Y_RANGE $SECTION_RANGE $IS_FORCE 
# 
# ./run_pipeline.sh WS $Z_RANGE $X_RANGE $Y_RANGE $SECTION_RANGE $IS_FORCE 
# 
# ./run_pipeline.sh NP $Z_RANGE $X_RANGE $Y_RANGE $SECTION_RANGE $IS_FORCE 
# 
# ./run_pipeline.sh MERGE_PREPROCESS $Z_RANGE $X_RANGE $Y_RANGE $SECTION_RANGE $IS_FORCE 
#  
./run_pipeline.sh MERGE_BLOCK $Z_RANGE $X_RANGE $Y_RANGE $SECTION_RANGE $IS_FORCE 

./run_pipeline.sh MERGE_COMBINE $Z_RANGE $X_RANGE $Y_RANGE $SECTION_RANGE $IS_FORCE 

./run_pipeline.sh MERGE_RELABEL $Z_RANGE $X_RANGE $Y_RANGE $SECTION_RANGE $IS_FORCE 

./run_pipeline.sh DEBUG_GENERATE $Z_RANGE $X_RANGE $Y_RANGE $SECTION_RANGE $IS_FORCE 

