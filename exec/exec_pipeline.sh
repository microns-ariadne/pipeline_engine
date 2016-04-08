#!/bin/bash

#./run_pipeline.sh K11-S1-AC3-256-sub-sample-2 ./K11_S1/labels_928x928 ./K11_S1/probs_512x512_up_2

#./run_pipeline.sh K11-S1-AC3-256-L-100 ./K11_S1/labels_100 ./K11_S1/probs_100

#./run_pipeline.sh K11-S1-AC3-256-2D-ws-seeds-dist-4-only-0 ./K11_S1/labels_976x976 ./K11_S1/probs_53_dist_4_only_0

#./run_pipeline.sh K11-S1-AC3-256-dist-4 ./K11_S1/labels_976x976 ./K11_S1/probs_53_dist_4_combined

#./run_pipeline.sh K11-S1-AC3-256-w2-AC4-train ./K11_S1/labels_972x972_3D ./K11_S1/probs_AC4_train

./run_pipeline.sh AC3-test-QTD-64 ./QTD/labels_test_972x972 ./QTD/output_QTD_test_64/1

#./run_pipeline.sh AC3-train-QTD-64 ./QTD/labels_train_972x972 ./QTD/output_QTD_train_64/1

#./run_pipeline.sh ac3-test-dist-4-3D-orig ./AC3/new/labels_test_976x976 ./AC3/new/probs

#./run_pipeline.sh AC3-TEST-NEW-SEEDS-2 ./AC3/new/labels_test_976x976 ./AC3/new/prob-distSeeds

#./run_pipeline.sh AC3-TRAIN-Z-EXTEND ./AC3/z-extend/labels_972x972 ./AC3/z-extend/probs

# ./run_pipeline.sh P3-block-0-4-4-segs-2 ./probs-block-0-4-4-segments/segments/2
# ./run_pipeline.sh P3-block-0-4-4-segs-3 ./probs-block-0-4-4-segments/segments/3
# ./run_pipeline.sh P3-block-0-4-4-segs-4 ./probs-block-0-4-4-segments/segments/4
# ./run_pipeline.sh P3-block-0-4-4-segs-5 ./probs-block-0-4-4-segments/segments/5
# ./run_pipeline.sh P3-block-0-4-4-segs-6 ./probs-block-0-4-4-segments/segments/6
# ./run_pipeline.sh P3-block-0-4-4-segs-7 ./probs-block-0-4-4-segments/segments/7
# ./run_pipeline.sh P3-block-0-4-4-segs-8 ./probs-block-0-4-4-segments/segments/8
# ./run_pipeline.sh P3-block-0-4-4-segs-9 ./probs-block-0-4-4-segments/segments/9