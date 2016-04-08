#!/bin/bash

# BORDER_WIDTH=5
# CLOSE_WIDTH=13
# INPUT=/mnt/disk3/armafire/datasets/K11_S1_debug/K11_S1_1024x1024x100_data/block_0000_0003_0001/block_id_0000_0003_0001_segment_0000_range_0000_0099_type_normal
# OUTPUT=./output
# 
# echo matlab -nodisplay -nojvm  -r "border_masks $BORDER_WIDTH $CLOSE_WIDTH $INPUT $OUTPUT; quit;"
# matlab -nodisplay -nojvm  -r "border_masks $BORDER_WIDTH $CLOSE_WIDTH $INPUT $OUTPUT; quit;"

echo matlab -nodisplay -nojvm  -r "border_masks $1 $2 $3 $4; quit;"
matlab -nodisplay -nojvm  -r "border_masks $1 $2 $3 $4; quit;"
