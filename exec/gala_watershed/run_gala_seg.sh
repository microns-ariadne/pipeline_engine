#!/bin/bash


MEMBRANES_PROBS_H5=../results/isbi_fully_conv_17x17_w2_8f/probs_1008x1008/h5-stacked-probs-fully-conv-17-17-w2-8f-1008-1008.h5

echo "PROBS: $MEMBRANES_PROBS_H5"
echo "python gala_overseg.py \
	--pixelprob-file=$MEMBRANES_PROBS_H5 \
	--seed-size 5 \
	. "
	
python gala_overseg.py \
	--pixelprob-file=$MEMBRANES_PROBS_H5 \
	--seed-size 5 \
	. \

