#!/bin/sh

./build/neuroproof_graph_predict \
    /home/vj/neuroproof/data/isbi/supervoxels.h5 \
    /home/vj/neuroproof/data/isbi/train-input.tif  \
    /home/vj/neuroproof/data/isbi/classifier.xml \
    --num-top-edges 256 --agglo-type 3

#./build/neuroproof_graph_predict \
#    --classifier-file /home/vj/neuroproof/data/isbi/classifier.xml \
#    --prediction-file /home/vj/neuroproof/data/isbi/train-input.tif  \
#    --watershed-file /home/vj/neuroproof/data/isbi/supervoxels.h5 \
#    --num-top-edges 256 --agglo-type 5
