#!/bin/bash
export CILK_NWORKERS=$1
/scratch/neuro_segmentation/NeuroProof/build/bin/neuroproof_graph_predict --agglo-type 5 --num-top-edges $2 --rand-prior $3  supervoxels.h5  ISBI_data/train-membranes-idsia.h5 classifier.xml
