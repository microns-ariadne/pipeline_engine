#!/bin/bash

#INPUT=/mnt/disk3/armafire/datasets/K11/blocks/K11_2048x2048x100_em/block_0000_0004_0004
#OUTPUT=/mnt/disk3/armafire/datasets/K11/blocks/K11_2048x2048x100_em_cc/block_0000_0004_0004

echo matlab -nodisplay -nojvm  -r "cc_block $1 $2; quit;"

matlab -nodisplay -nojvm  -r "cc_block $1 $2; quit;"
