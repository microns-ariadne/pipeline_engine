#!/bin/bash

# CNN_CROP_LEG_STR=26
# IN_OUTER_MASK_FILENAME=/mnt/disk3/armafire/datasets/K11_S1_debug/K11_S1_1024x1024x100_meta/all_blocks_0003_0001_meta/blocks_0003_0001_outer_mask_total.png
# IN_BORDER_MASK_FILENAME=/mnt/disk3/armafire/datasets/K11_S1_debug/K11_S1_1024x1024x100_meta/all_blocks_0003_0001_meta/blocks_0003_0001_border_mask_total.png
# IN_BLOCKPATH_META=/mnt/disk3/armafire/datasets/K11_S1_debug/K11_S1_1024x1024x100_meta/block_0000_0003_0001_meta/block_id_0000_0003_0001_segment_0000_range_0000_0099_type_normal_meta
# IN_BLOCKPATH_PROBS=/mnt/disk3/armafire/datasets/K11_S1_debug/K11_S1_1024x1024x100_probs_combined/block_0000_0003_0001_probs/block_id_0000_0003_0001_segment_0000_range_0000_0099_type_normal_probs
# OUT_BLOCKPATH_WS=./output/ws
# OUT_BLOCKPATH_NP=./output/np
# 
# echo matlab -nodisplay -nojvm  -r "ws_np_prepare $CNN_CROP_LEG_STR $IN_OUTER_MASK_FILENAME $IN_BORDER_MASK_FILENAME $IN_BLOCKPATH_META $IN_BLOCKPATH_PROBS $OUT_BLOCKPATH_WS $OUT_BLOCKPATH_NP; quit;"
# matlab -nodisplay -nojvm  -r "ws_np_prepare $CNN_CROP_LEG_STR $IN_OUTER_MASK_FILENAME $IN_BORDER_MASK_FILENAME $IN_BLOCKPATH_META $IN_BLOCKPATH_PROBS $OUT_BLOCKPATH_WS $OUT_BLOCKPATH_NP; quit;"

echo matlab -nodisplay -nojvm  -r "ws_np_prepare $1 $2 $3 $4 $5 $6 $7; quit;"
matlab -nodisplay -nojvm  -r "ws_np_prepare $1 $2 $3 $4 $5 $6 $7; quit;"

