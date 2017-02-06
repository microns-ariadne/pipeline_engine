#!/bin/bash

for value in {0..99}
do
CMD="python ./scripts/bin_to_tiff.py ./test_49_w2_4M_8f_16f_32f/probs/$(printf "%04d" $value)-probs-isbi-49-w2-4M-8f.bin ./test_49_w2_4M_8f_16f_32f/probs_tif/$(printf "%04d" $value)-probs-isbi-49-w2-4M-8f"
echo $CMD
$CMD
done

