The merge_pair.py script proves interface for merging two blocks.

-----------------------------
Hard coded parameters (see util.py for modifying and viewing)
DATA_DIR          : Root directory of block segmentations and probability maps
NP_PREDICT        : Path to NeuroProof prediction exe
SLICE_CLASSIFIERS : Pre-generated classifiers for merging slices
TEMPLATE FNAMES   : Template names of files to be read or generated
HEURISTIC THRSHLD : Thresholds for determining merges heuristically. One number for each direction.
Z                 : The depth of ONE block (typically 100).

-----------------------------
Input:
    --work-dir  : A directory where all temporary and resulting files will be stored.
                  In case of running many pairs concurrently, this directory can be used for all,
                  because files will have distinct names.
    --block     : A triple of block index in the order given in files in DATA_DIR (K11 for example).
    --dir       : The axis to merge in (0, 1 or 2).
    --width     : The width of the slice.

Optional Arguments:
    --force                 : Rewrite files [default: not set]
    --classifier CLASSIFIER : Use the given one instead of default
    --np-predict NP_PREDICT : Path to segmentation executable.
    --save-blocks***        : Whether to save the retrieved blocks [default: not set]
    --np-args               : Arguments to be passed to NP_PREDICT
    --gen-ws-slice          : Generate WS slice (for testing) [default: not set]

-------------------------------
Product for block (Z, Y, X) and direction D (in directory work-dir):
  *** IF save-blocks is set
    probs_Z_Y_X.h5, segmentation_Z_Y_X.h5       : probability and segmentation for block (Z, Y, X)
    probs_Z'_Y'_X'.h5, segmentation_Z'_Y'_X'.h5 : probability and segmentation for block (Z', Y', X')
                                                  that is neighbouring to (Z, Y, X) in D direction

    slice_supv_Z_Y_X_D.h5         : slice supervoxels between (Z, Y, X) and (Z', Y', X')
    slice_probs_Z_Y_X_D.h5        : slice probabilities between (Z, Y, X) and (Z', Y', X')
    slice_segmentation_Z_Y_X_D.h5 : slice segmentation between (Z, Y, X) and (Z', Y', X')

    merge_Z_Y_X_D.txt : a file that contains a list of pairs (l1, l2) of labels, which means that
                        label l1 from first block should be merged to label l2 of other block.
                        The file has two parts
                            (1) Merge pairs decided by running NP_PREDICT on slice
                            (2) Heuristic merge pairs. A label (l1, l2) will be merged if they 
                                have a common border along the cut of more than threshold pixels.

-------------------------------

***NOTE: When running in parallel on many merge pairs when --save-block option is on,
         two instances of merge_pair.py might try to write to the same files
         (probs_*_*_*.h5 and segmentation_*_*_*.h5).

         To avoid such behaviour, one can distribute the computation in two stages
            (1) Run all merges for EVEN/ODD Z and D = 0 concurrently
                    (note, pay attention to the parity of the number of blocks in Z direction).
                As a result all files that could have been written concurrently will be written to disk.
            (2) Run all merge pairs that are left, concurrently.

