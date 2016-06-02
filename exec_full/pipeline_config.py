import os
import sys

###############################################################################
# Align config
###############################################################################

ALIGN_RES_TILE_ROWS = 30000
ALIGN_RES_TILE_COLS = 30000

ALIGN_RES_MIN_FILE_ID = 0
ALIGN_RES_MAX_FILE_ID = 1849

ALIGN_RES_FILE_TEMPLATE = 'res-full-warped-sec-%.4d.tif'

###############################################################################
# Block config
###############################################################################
BLOCK_N_IMAGES = 100
BLOCK_N_ROWS = 2048
BLOCK_N_COLS = 2048

BLOCKS_MIN_DEPTH = 0
BLOCKS_MAX_DEPTH = int(ALIGN_RES_MAX_FILE_ID / BLOCK_N_IMAGES)

BLOCK_SUB_SAMPLE = 2

BLOCKS_MIN_ROW_ID = 0
BLOCKS_MIN_COL_ID = 0

BLOCKS_MAX_ROW_ID = 14
BLOCKS_MAX_COL_ID = 14

CNN_IS_3D_FIX = 1
CNN_DEPTH_LEG = 1
CNN_PATCH_LEG = 52

Z_PAD = 10
X_PAD = 60
Y_PAD = 60

if (CNN_IS_3D_FIX) and (Z_PAD == 0):
    Z_PAD = 1
    
PIPELINE_CNN_PATCH_LEG = 26

PREFIX = 'K11_S1_3nm'

###############################################################################
# Dir config
###############################################################################

META_DIR = '/mnt/disk1/armafire/datasets/K11_S1_3nm_meta/'

DATA_DIR_LIST = [
    '/mnt/disk1/armafire/datasets/K11_S1_3nm_data_1/',
    '/mnt/disk2/armafire/datasets/K11_S1_3nm_data_2/',
    '/mnt/disk3/armafire/datasets/K11_S1_3nm_data_3/',
    '/mnt/disk4/armafire/datasets/K11_S1_3nm_data_4/',
    '/mnt/disk5/armafire/datasets/K11_S1_3nm_data_5/',
]

ALIGN_DIR = 'align'
ALIGN_WORK_DIR = 'work_dir'
ALIGN_RESULT_DIR = 'result_dir'
BLOCKS_META_DIR = 'blocks_meta'
EM_DIR = 'em_padded'
PROBS_DIR = 'probs'
PROBS_WS_DIR = 'probs_ws'
PROBS_NP_DIR = 'probs_np'
WS_DIR = 'ws'
NP_DIR = 'np'
MERGE_DIR = 'merge'

SCATTER_POINTS_DIR = 'scatter_points'
SKELETONS_DIR = 'skeletons'
LOG_DIR = 'log'
LOG_PROCS_DIR = 'log_procs'
LOG_RUNS_DIR = 'log_runs'

DEBUG_DIR = 'debug'

###############################################################################
# Meta config
###############################################################################

META_BLOCK_STATUS_FILE = 'block_%.4d_%.4d_%.4d_status.txt'
META_BLOCK_STATUS_NOT_VALID = 'NOT-VALID'
META_BLOCK_STATUS_VALID = 'VALID'

###############################################################################
# GENERATE_BLOCKS params
###############################################################################
GENERATE_BLOCKS_SECTION_BATCH = 4

GENERATE_BLOCKS_BIN_PATH = 'python /home/armafire/Pipeline/pipeline_engine/exec_full/generate_blocks.py'

N_GENERATE_BLOCKS_WORKERS_PER_CPU = 12

###############################################################################
# ANALYZE_BLOCKS params
###############################################################################
ANALYZE_BLOCKS_BIN_PATH = 'python /home/armafire/Pipeline/pipeline_engine/exec_full/analyze_blocks.py'

N_ANALYZE_BLOCKS_WORKERS_PER_CPU = 9 * 2

###############################################################################
# CNN params
###############################################################################

N_FC_DNN_WORKERS_PER_CPU = 4
N_PER_FC_DNN_WORKERS = 9

FC_DNN_PATH = '/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/build_pipeline/run_dnn_K11_S1_3nm_w2_bg_32f_105x105_3D_4M_K11_3nm_AC3'

FC_DNN_3D_DEPTH = 3

FC_DNN_N_CHANNELS = 2

###############################################################################
# PREPARE WS_NP_PREPARE params
###############################################################################

WS_NP_PREPARE_PATH = 'python /home/armafire/Pipeline/pipeline_engine/exec_full/ws_np_prepare.py'

N_WS_NP_PREPARE_WORKERS_PER_CPU = 8 * 2

###############################################################################
# WS params
###############################################################################

N_WS_WORKERS_PER_CPU = 8 * 4

WS_PATH = '/home/armafire/Pipeline/pipeline_engine/watershed/build/watershed.x'

IS_WS_SEEDS_2D = False
IS_WS_2D = False
IS_WS_USE_BG = True

###############################################################################
# NP config
###############################################################################

NP_THRESHOLD = 0.1

N_NP_EXEC_WORKERS_PER_CPU = 8 * 2

N_PER_NP_EXEC_CILK_WORKERS = 4

NP_EXEC_PATH = 'python /home/armafire/Pipeline/pipeline_engine/exec_full/np_exec.py'

NP_BIN_PATH = '/home/armafire/Pipeline/pipeline_engine/neuroproof/MIT_agg/MIT_agg/neuroproof_agg/npclean/build/neuroproof_graph_predict'

NP_CLASSIFIER_DIR = '/home/armafire/Pipeline/pipeline_engine/neuroproof/np_classifiers/np-classifier-K11-3nm-AC3-train-w2-bg-32f-105x105-3D-sub-2-ws-v0-s5'

#'/home/armafire/Pipeline/pipeline_engine/neuroproof/np_classifiers/np-classifier-K11-3nm-AC3-100_105x105-3D-WS'

#'/home/armafire/Pipeline/pipeline_engine/neuroproof/np_classifiers/np-classifier-K11-3nm-AC3-256_105x105-3D-WS-w2-bg-sub-2'

#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-K11-S1-AC3-256-53-dist-4-GT/'

#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-K11-AC-256-3D-PAD-FM25/'
#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-P3-errors-3D-49x49-32f-GT/' 
#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-K11-AC-256-3D-PAD-FM25/'
#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-P3-errors-NEW-2-3D-49x49-32f-GT-ws-2D'
#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-P3-errors-NEW-2-3D-49x49-32f-GT-ws-2D'
#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-P3-errors-NEW-2-3D-49x49-32f-GT/'

###############################################################################
# MERGE common params
###############################################################################

MERGE_BASE_DIR = '/home/armafire/Pipeline/pipeline_engine/merge/np-merge'

###############################################################################
# MERGE_PREPROCESS params
###############################################################################

MERGE_PREPROCESS_PATH = os.path.join(MERGE_BASE_DIR, 'python/run_preprocess.sh')

MERGE_PREPROCESS_Z_OVERLAP = 1

N_MERGE_PREPROCESS_WORKERS_PER_CPU = 8 * 2

N_MERGE_PREPROCESS_EXEC_CILK_WORKERS_PER_RUN = 4

###############################################################################
# MERGE_BLOCK_EXEC params
###############################################################################

MERGE_PAIR_BIN_PATH = 'python ' + os.path.join(MERGE_BASE_DIR, 'python/merge_pair.py')

MERGE_BLOCK_EXEC_PATH = 'python ' + os.path.join(MERGE_BASE_DIR, 'python/run_block_merge.py')

N_MERGE_BLOCK_EXEC_WORKERS_PER_CPU = 8 * 2

N_MERGE_BLOCK_EXEC_CILK_WORKERS_PER_RUN = 4

MERGE_BLOCK_NP_BINARY = NP_BIN_PATH

MERGE_BLOCK_NP_THRESHOLD_PARAM = '--threshold=0.05'

###############################################################################
# MERGE_COMBINE params
###############################################################################

MERGE_COMBINE_BIN_PATH = os.path.join(MERGE_BASE_DIR, 'cpp/combine')

MERGE_COMBINE_MAXIDS_BIN_PATH = 'python ' + os.path.join(MERGE_BASE_DIR, 'python/combine_maxIDs.py')

MERGE_COMBINE_EXEC_PATH = 'python ' + os.path.join(MERGE_BASE_DIR, 'cpp/run_combine.py')

###############################################################################
# MERGE_RELABEL params
###############################################################################

MERGE_RELABEL_EXEC_PATH = os.path.join(MERGE_BASE_DIR, 'cpp/run_relabel.sh')

N_MERGE_RELABEL_WORKERS_PER_CPU = 32


###############################################################################
# Exec config
###############################################################################

PROC_SUCCESS_STR = '-= PROC SUCCESS =-'
N_CPUS = 4
