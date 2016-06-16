#!/bin/bash

# INPUT=/home/armafire/Pipeline/exec/K11_S1/probs_53_dist_4
# OUTPUT=/home/armafire/Pipeline/exec/K11_S1/probs_53_dist_4_combined
# KERNEL_SIZE=13
# REGIONAL_DEPTH_REDUCE=0.04
# STD_SMOOTH=2
# IM_SIZE_D1=976
# IM_SIZE_D2=976
# N_IMAGES=256

#echo matlab -nodisplay -r "extractSeeds $INPUT $OUTPUT $KERNEL_SIZE $REGIONAL_DEPTH_REDUCE $STD_SMOOTH $IM_SIZE_D1 $IM_SIZE_D2 $N_IMAGES; quit;"
#matlab -nodisplay -r "extractSeeds $INPUT $OUTPUT $KERNEL_SIZE $REGIONAL_DEPTH_REDUCE $STD_SMOOTH $IM_SIZE_D1 $IM_SIZE_D2 $N_IMAGES; quit;"

echo matlab -nodisplay -r "extractSeeds $1 $2 $3 $4 $5 $6 $7 $8; quit;"
matlab -nodisplay -r "extractSeeds $1 $2 $3 $4 $5 $6 $7 $8; quit;"
