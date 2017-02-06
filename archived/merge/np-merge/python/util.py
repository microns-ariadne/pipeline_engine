import os
import time

import numpy as np
from pipeline_common import *

###############################################################################
# Paths
###############################################################################
SEGM_DIR  = "{dir:s}/block_{z:04d}_{y:04d}_{x:04d}_np"
PROBS_DIR = "{dir:s}/block_{z:04d}_{y:04d}_{x:04d}_probs"

# CHUNK_TEMP = '{data}_{z:04d}_{y:04d}_{x:04d}.h5'
CHUNK_TEMP_DIR = '{data}_{z:04d}_{y:04d}_{x:04d}'

# SLICE_TEMP = 'slice_{data}_{z:04d}_{y:04d}_{x:04d}_dir_{d}_width_{w}.h5'
SLICE_TEMP_DIR = 'slice_{data}_{z:04d}_{y:04d}_{x:04d}_dir_{d}_width_{w}'

MERGE_PAIR_TEMP = 'merge_{z:04d}_{y:04d}_{x:04d}_dir_{d}.txt'

###############################################################################
# Watershed
###############################################################################

# WATERSHED = '/mnt/disk1/hayk/oversegmentation/cpp/build/watershed.x'
# WATERSHED = '/mnt/disk1/hayk/oversegmentation-h5-make-0/watershed-3D-seeds-3D-BFS-0-BG-HDF5.x'
# WATERSHED = '/mnt/disk1/hayk/execs/watershed-3D-3D_seeds-255_TO_0-RGB.x'
WATERSHED = '/mnt/disk1/hayk/execs/watershed-3D-seeds-3D-BFS-0-BG-RGB.x'

###############################################################################
# Neuroproof
###############################################################################

NP_PREDICT = NP_BIN_PATH

# NP_PREDICT = '/home/armafire/Pipeline/exec/neuroproof_agg/npclean/build/neuroproof_graph_predict'
#NP_PREDICT = '/mnt/disk1/hayk/MIT_agg/neuroproof_agg/npclean/build/neuroproof_graph_predict'
# NP_PREDICT = '/mnt/disk1/hayk/execs_from_cnx1/npclean/build/neuroproof_graph_predict'

# CLASSIFIER = '/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-K11-AC-256-3D-PAD/np-classifier-K11-S1-AC3-256.xml' # general classifier
# CLASSIFIER = '/mnt/disk1/hayk/3nm-general-classifier/np-classifier-K11-3nm-AC3-100_105x105-3D-WS:np-classifier-K11-3nm-AC3-100_105x105-3D-WS.xml' # 3nm
CLASSIFIER = NP_CLASSIFIER_DIR 
#'/mnt/disk4/armafire/Pipeline/exec/results-K11-6nm-AC3-and-AC4-dist-4-32f-53x53-3D-blacked-ws-v0-s5/np-classifier/np-classifier-K11-6nm-AC3-and-AC4-dist-4-32f-53x53-3D-blacked-ws-v0-s5.xml'

# classifiers with 8 cuts
# SLICE_CLASSIFIERS = [CLASSIFIER, CLASSIFIER, CLASSIFIER]
SLICE_CLASSIFIERS = [
        '/home/armafire/Pipeline/np-classifiers/merge-classifiers/6nm/ac3-dir0/slice-classifier-0-32-8.xml',
        '/home/armafire/Pipeline/np-classifiers/merge-classifiers/6nm/ac3-dir1/slice-classifier-1-128-8.xml',
        '/home/armafire/Pipeline/np-classifiers/merge-classifiers/6nm/ac3-dir2/slice-classifier-2-128-8.xml',
#    os.path.join(NP_CLASSIFIER_DIR, 'np-classifier-K11-3nm-AC3-train-w2-bg-32f-105x105-3D-sub-2-ws-v0-s5.xml'),
#    os.path.join(NP_CLASSIFIER_DIR, 'np-classifier-K11-3nm-AC3-train-w2-bg-32f-105x105-3D-sub-2-ws-v0-s5.xml'),
#    os.path.join(NP_CLASSIFIER_DIR, 'np-classifier-K11-3nm-AC3-train-w2-bg-32f-105x105-3D-sub-2-ws-v0-s5.xml'),
#        '/mnt/disk1/hayk/merge-classifiers/ac3-thick-6nm/slice-classifier-0-32-8.xml',
#        '/mnt/disk1/hayk/merge-classifiers/ac3-thick-6nm/slice-classifier-1-128-8.xml',
#        '/mnt/disk1/hayk/merge-classifiers/ac3-thick-6nm/slice-classifier-2-128-8.xml'
]

###############################################################################
# VI
###############################################################################
HEURISTIC_THRESHOLD = np.array([2500, 500, 500], dtype='int')


###############################################################################
# Common funcs
###############################################################################

def chunk_dir_path(
    work_dir, 
    (blockZ, blockY, blockX), 
    data_type, 
    alwaysReturn=False):
    
    path = os.path.join(
        work_dir, 
        CHUNK_TEMP_DIR.format(
            data=data_type, 
            z=blockZ, y=blockY, x=blockX))
        
    if alwaysReturn or os.path.exists(path):
        return path
        
    return None

def slice_dir(
    args,
    merge_dir,
    data_type):
    return os.path.join(
        merge_dir, 
        SLICE_TEMP_DIR.format(
            data=data_type, 
            x=args.block[2], y=args.block[1], z=args.block[0], 
            d=args.direction, w=args.width))

def merge_path(args):
    return os.path.join(
        args.meta_dir, 
        MERGE_PAIR_TEMP.format(
            x=args.block[2], y=args.block[1], z=args.block[0], 
            d=args.direction))

