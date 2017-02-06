#!/bin/bash

#WORK_DIR=/mnt/disk3/armafire/datasets/K11_S1/blocks_full/K11_S1_1024x1024x100_np_merge

IS_FORCE="--force"

cd "$(dirname "$0")"

./relabel \
    $IS_FORCE \
    --meta-dir $1 \
    --block-dir $2 \
    --block $3 $4 $5


