#!/bin/bash

# INPUT_NP=/mnt/disk3/armafire/datasets/K11_S1/blocks/K11_S1_1024x1024x100_np
# BLOCK=block_0000_0002_0002*
# Y_DIM=1024
# X_DIM=1024
# DEPTH=100
# IS_VIDEO=1
# INPUT_EM=/mnt/disk3/armafire/datasets/K11_S1/blocks/K11_S1_1024x1024x100_em_cc

#echo matlab -nodisplay -r "mergeH5_1core $INPUT_NP $BLOCK $Y_DIM $X_DIM $DEPTH $IS_VIDEO $INPUT_EM; quit;"

echo matlab -nodisplay -r "mergeH5_1core $1 $2 $3 $4 $5 $6 $7; quit;"
matlab -nodisplay -r "mergeH5_1core $1 $2 $3 $4 $5 $6 $7; quit;"

