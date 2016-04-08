#!/bin/bash

#####################################################################
# Params
#####################################################################
PHASE_ALIGN_0_GENERATE_TILES=ALIGN_0_GENERATE_TILES # 1
PHASE_ALIGN_GENERATE_TILES_TXT=ALIGN_GENERATE_TILES_TXT # 2
PHASE_ALIGN_1_COMPUTE_KPS_AND_MATCH=ALIGN_1_COMPUTE_KPS_AND_MATCH # 3
PHASE_ALIGN_2_COMPUTE_TRANSFORMS=ALIGN_2_COMPUTE_TRANSFORMS # 4
PHASE_ALIGN_3_COMPUTE_WARPS=ALIGN_3_COMPUTE_WARPS # 5

PHASE_PREPARE_DATA_CC=PREPARE_DATA_CC # 6
PHASE_PREPARE_DATA_BLACKEN=PREPARE_DATA_BLACKEN # 7
PHASE_PREPARE_DATA_SUBBLOCKS=PREPARE_DATA_SUBBLOCKS # 8
PHASE_BORDER_MASKS=BORDER_MASKS # 9
PHASE_COMBINE_BORDER_MASKS=COMBINE_BORDER_MASKS # 10
PHASE_CNN=CNN # 11
PHASE_PROBS_COMBINE=PROBS_COMBINE # 12
PHASE_WS_NP_PREPARE=WS_NP_PREPARE # 13
PHASE_WS=WS # 14
PHASE_NP_PREPARE=NP_PREPARE # 15
PHASE_NP_EXEC=NP_EXEC # 16
PHASE_BLOCK_VIDEO=BLOCK_VIDEO # 17
PHASE_MERGE_PREPROCESS=MERGE_PREPROCESS # 18
PHASE_MERGE_EXEC=MERGE_EXEC # 19
PHASE_MERGE_COMBINE=MERGE_COMBINE # 20
PHASE_MERGE_RELABEL=MERGE_RELABEL # 21
PHASE_SKELETONS=SKELETONS # 22
PHASE_DEBUG_GENERATE=DEBUG_GENERATE # 23

#####################################################################
# Env setup
#####################################################################

source ./init.sh

#####################################################################
# Exec
#####################################################################

if [ $# -eq 0 ]; then

#echo "python execute_pipeline_full.py $PHASE_PREPARE_DATA_CC"

echo "python execute_pipeline_full.py ALL"

# python execute_pipeline_full.py $PHASE_PREPARE_DATA_CC
# # python execute_pipeline_full.py $PHASE_PREPARE_DATA_BLACKEN
# python execute_pipeline_full.py $PHASE_PREPARE_DATA_SUBBLOCKS
# python execute_pipeline_full.py $PHASE_CNN
# python execute_pipeline_full.py $PHASE_WS_NP_PREPARE
# python execute_pipeline_full.py $PHASE_WS
# python execute_pipeline_full.py $PHASE_NP_PREPARE
# python execute_pipeline_full.py $NP_EXEC

# echo "python execute_pipeline_full.py 1 > full_run_1.log"
# python execute_pipeline_full.py 1 > full_run_1.log
# echo "python execute_pipeline_full.py 2 > full_run_2.log"
# python execute_pipeline_full.py 2 > full_run_2.log
# echo "python execute_pipeline_full.py 3 > full_run_3.log"
# python execute_pipeline_full.py 3 > full_run_3.log
# echo "python execute_pipeline_full.py 4 > full_run_4.log"
# python execute_pipeline_full.py 4 > full_run_4.log
# echo "python execute_pipeline_full.py 5 > full_run_5.log"
# python execute_pipeline_full.py 5 > full_run_5.log
# echo "python execute_pipeline_full.py 6 > full_run_6.log"
# python execute_pipeline_full.py 6 > full_run_6.log
# echo "python execute_pipeline_full.py 7 > full_run_7.log"
# python execute_pipeline_full.py 7 > full_run_7.log
# echo "python execute_pipeline_full.py 8 > full_run_8.log"
# python execute_pipeline_full.py 8 > full_run_8.log
# echo "python execute_pipeline_full.py 9 > full_run_9.log"
# python execute_pipeline_full.py 9 > full_run_9.log
# echo "python execute_pipeline_full.py 10 > full_run_10.log"
# python execute_pipeline_full.py 10 > full_run_10.log
# echo "python execute_pipeline_full.py 11 > full_run_11.log"
# python execute_pipeline_full.py 11 > full_run_11.log
# echo "python execute_pipeline_full.py 12 > full_run_12.log"
# python execute_pipeline_full.py 12 > full_run_12.log
# echo "python execute_pipeline_full.py 13 > full_run_13.log"
# python execute_pipeline_full.py 13 > full_run_13.log
# echo "python execute_pipeline_full.py 14 > full_run_14.log"
# python execute_pipeline_full.py 14 > full_run_14.log
# echo "python execute_pipeline_full.py 15 > full_run_15.log"
# python execute_pipeline_full.py 15 > full_run_15.log
# echo "python execute_pipeline_full.py 16 > full_run_16.log"
# python execute_pipeline_full.py 16 > full_run_16.log
# echo "python execute_pipeline_full.py 17 > full_run_17.log"
# python execute_pipeline_full.py 17 > full_run_17.log
# echo "python execute_pipeline_full.py 18 > full_run_18.log"
# python execute_pipeline_full.py 18 > full_run_18.log
# echo "python execute_pipeline_full.py 19 > full_run_19.log"
# python execute_pipeline_full.py 19 > full_run_19.log
# echo "python execute_pipeline_full.py 20 > full_run_20.log"
# python execute_pipeline_full.py 20 > full_run_20.log
# echo "python execute_pipeline_full.py 21 > full_run_21.log"
# python execute_pipeline_full.py 21 > full_run_21.log
echo "python execute_pipeline_full.py 22 > full_run_22.log"
python execute_pipeline_full.py 22 > full_run_22.log
echo "python execute_pipeline_full.py 23 > full_run_23.log"
python execute_pipeline_full.py 23 > full_run_23.log

else

echo "python execute_pipeline_full.py $1"

python execute_pipeline_full.py $1

fi


