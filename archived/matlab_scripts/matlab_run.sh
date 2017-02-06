#!/bin/bash

echo matlab -nodisplay -nojvm  -r 'mergeH5_1core /mnt/disk1/armafire/datasets/P3/blocks_tile_1_1/tile_1_1_2048x2048x100_np block_0000_0004_0004*; exit;'

matlab -nodisplay -nojvm  -r 'mergeH5_1core /mnt/disk1/armafire/datasets/P3/blocks_tile_1_1/tile_1_1_2048x2048x100_np block_0000_0004_0004*; exit;'