
import sys
import os
import argparse
import shutil
import time
import subprocess
from imread import imread,imsave

import cv2
import numpy as np

# General params

PHASE_ALIGN_GENERATE_TILES = 'ALIGN_GENERATE_TILES'
PHASE_ALIGN_GENERATE_TILES_TXT = 'ALIGN_GENERATE_TILES_TXT'
PHASE_ALIGN_COMPUTE_KPS_AND_MATCH = 'ALIGN_COMPUTE_KPS_AND_MATCH'
PHASE_ALIGN_COMPUTE_TRANSFORMS = 'ALIGN_COMPUTE_TRANSFORMS'
PHASE_ALIGN_COMPUTE_WARPS = 'ALIGN_COMPUTE_WARPS'

PHASE_PREPARE_DATA_CC = 'PREPARE_DATA_CC'
PHASE_PREPARE_DATA_BLACKEN = 'PREPARE_DATA_BLACKEN'
PHASE_PREPARE_DATA_SUBBLOCKS = 'PREPARE_DATA_SUBBLOCKS'
PHASE_BORDER_MASKS = 'BORDER_MASKS'
PHASE_COMBINE_BORDER_MASKS = 'COMBINE_BORDER_MASKS'
PHASE_CNN = 'CNN'
PHASE_PROBS_COMBINE = 'PROBS_COMBINE'
PHASE_WS_NP_PREPARE = 'WS_NP_PREPARE'
PHASE_WS = 'WS'
PHASE_NP_PREPARE = 'NP_PREPARE'
PHASE_NP_EXEC = 'NP_EXEC'
PHASE_BLOCK_VIDEO = 'BLOCK_VIDEO'
PHASE_MERGE_PREPROCESS = 'MERGE_PREPROCESS'
PHASE_MERGE_EXEC = 'MERGE_EXEC'
PHASE_MERGE_COMBINE = 'MERGE_COMBINE'
PHASE_MERGE_RELABEL = 'MERGE_RELABEL'
PHASE_SKELETONS = 'SKELETONS'
PHASE_DEBUG_GENERATE = 'DEBUG_GENERATE'

PHASE_LIST = [
    
    PHASE_ALIGN_GENERATE_TILES,
    PHASE_ALIGN_GENERATE_TILES_TXT,
    PHASE_ALIGN_COMPUTE_KPS_AND_MATCH,
    PHASE_ALIGN_COMPUTE_TRANSFORMS,
    PHASE_ALIGN_COMPUTE_WARPS,
    
    PHASE_PREPARE_DATA_CC,
    PHASE_PREPARE_DATA_BLACKEN,
    PHASE_PREPARE_DATA_SUBBLOCKS,
    PHASE_BORDER_MASKS,
    PHASE_COMBINE_BORDER_MASKS,
    PHASE_CNN,
    PHASE_PROBS_COMBINE,
    PHASE_WS_NP_PREPARE,
    PHASE_WS,
    PHASE_NP_PREPARE,
    PHASE_NP_EXEC,
    PHASE_BLOCK_VIDEO,
    PHASE_MERGE_PREPROCESS,
    PHASE_MERGE_EXEC,
    PHASE_MERGE_COMBINE,
    PHASE_MERGE_RELABEL,
    PHASE_SKELETONS,
    PHASE_DEBUG_GENERATE,
]

N_CPUS = 4

IS_FLUSH_LOGS = 1

IS_WS_NP_PREPARE = 1
IS_WS_NP_PREPARE_USE_COMBINED_PROBS = 1

# Input params

BLOCK_N_DEPTH = 100
BLOCK_N_ROWS = 1024
BLOCK_N_COLS = 1024

BLOCKS_MAX_ROW_ID = 12
BLOCKS_MAX_COL_ID = 10

#BLOCKS_DIR = '/mnt/disk3/armafire/datasets/K11_S1_debug_full/'
BLOCKS_DIR = '/mnt/disk4/armafire/datasets/K11_S1_3nm/'

# ALL

BLOCKS_TO_PROCESS = [ (d,x,y) for d in xrange(0,18) for x in xrange(0,13) for y in xrange(0,10) ]

#BLOCKS_TO_PROCESS = [ (d,x,y) for d in xrange(0,4) for x in xrange(3,7) for y in xrange(3,7) ]

#BLOCKS_TO_PROCESS = [ (d,x,y) for d in xrange(0,4) for x in xrange(0,13) for y in xrange(0,10) ]

#BLOCKS_TO_PROCESS = [ (d,x,y) for d in xrange(0,1) for x in xrange(2,3) for y in xrange(2,3) ]

#BLOCKS_TO_PROCESS = [ (d,x,y) for d in xrange(0,1) for x in xrange(4,6) for y in xrange(4,6) ]

#BLOCKS_TO_PROCESS = [ (d,x,y) for d in xrange(0,4) for x in xrange(0,13) for y in xrange(0,10) ]

#BLOCKS_TO_PROCESS = [ (d,x,y) for d in xrange(0,2) for x in xrange(4,6) for y in xrange(4,6) ]

#BLOCKS_TO_PROCESS = [ (d,x,y) for d in xrange(0,4) for x in xrange(3,7) for y in xrange(3,7) ]

#BLOCKS_TO_PROCESS = [ (d,x,y) for d in xrange(0,2) for x in xrange(0,13) for y in xrange(0,10) ]

#BLOCKS_TO_PROCESS = [ (d,x,y) for d in xrange(0,2) for x in xrange(2,4) for y in xrange(2,4) ]


# BLOCKS_DIR_ALIGN = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_align')
# BLOCKS_DIR_EM = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_em')
# BLOCKS_DIR_EM_CC = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_em_cc')
# BLOCKS_DIR_INPUT = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_data')
# BLOCKS_DIR_META = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_meta')
# BLOCKS_DIR_PROBS = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_probs')
# BLOCKS_DIR_PROBS_COMBINED = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_probs_combined')
# BLOCKS_DIR_PROBS_WS = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_probs_ws')
# BLOCKS_DIR_PROBS_NP = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_probs_np')
# BLOCKS_DIR_WS = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_ws')
# BLOCKS_DIR_NP = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_np')
# BLOCKS_DIR_MERGE = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_merge')
# BLOCKS_DIR_SKELETONS = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_skeletons')
# BLOCKS_DIR_DEBUG = os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_debug')
# 
# BLOCKS_DIR_LOGS= os.path.join(BLOCKS_DIR, 'K11_S1_1024x1024x100_logs')

BLOCKS_DIR_ALIGN = os.path.join(BLOCKS_DIR, 'K11_S1_3nm_align')
BLOCKS_DIR_LOGS= os.path.join(BLOCKS_DIR, 'K11_S1_3nm_logs')

PHASE_ALIGN_GENERATE_TILES = 'ALIGN_GENERATE_TILES'
PHASE_ALIGN_GENERATE_TILES_TXT = 'ALIGN_GENERATE_TILES_TXT'
PHASE_ALIGN_COMPUTE_KPS_AND_MATCH = 'ALIGN_COMPUTE_KPS_AND_MATCH'
PHASE_ALIGN_COMPUTE_TRANSFORMS = 'ALIGN_COMPUTE_TRANSFORMS'
PHASE_ALIGN_COMPUTE_WARPS = 'ALIGN_COMPUTE_WARPS'

###############################################################################
# ALIGN general params
###############################################################################
N_ALIGN_BASE_SECTION = 40
N_ALIGN_SECTIONS = 16
N_ALIGN_BATCH = 4

ALIGN_PATH = '/home/armafire/Pipeline/align/align_proto/run_align'

BLOCKS_DIR_ALIGN_WORK_DIR = os.path.join(BLOCKS_DIR_ALIGN, 'work_dir')
BLOCKS_DIR_ALIGN_RESULT_DIR = os.path.join(BLOCKS_DIR_ALIGN, 'result_dir')

###############################################################################
# ALIGN_GENERATE_TILES params
###############################################################################
N_ALIGN_GENERATE_TILES_WORKERS_PER_CPU = 18
N_ALIGN_GENERATE_TILES_CPU_WORKERS = 1

ALIGN_EM_SOURCE = '/mnt/disk4/armafire/datasets/K11_orig_from_DanielBerger' 

ALIGN_TILES_PATH = 'python /home/armafire/Pipeline/align/align_proto/scripts/K11_S1_to_tiles.py'

BLOCKS_DIR_ALIGN_TILES = os.path.join(BLOCKS_DIR_ALIGN, 'tiles')

###############################################################################
# ALIGN_GENERATE_TILES_TXT params
###############################################################################

ALIGN_TILES_TXT_PATH = 'python /home/armafire/Pipeline/align/align_proto/scripts/K11_tiles_to_txt_input.py'

BLOCKS_DIR_ALIGN_TILES_TXT = os.path.join(BLOCKS_DIR_ALIGN, 'tiles_txt')

TILES_TXT_FILEPATH = os.path.join(
    BLOCKS_DIR_ALIGN_TILES_TXT, 
    'K11_S1_3nm_tiles_txt_%.4d-%.4d_0-4_0-4.txt' % (
        N_ALIGN_BASE_SECTION, 
        N_ALIGN_BASE_SECTION + N_ALIGN_SECTIONS))

###############################################################################
# ALIGN_COMPUTE_KPS_AND_MATCH params
###############################################################################

N_ALIGN_COMPUTE_KPS_AND_MATCH_WORKERS_PER_CPU = 1
N_ALIGN_COMPUTE_KPS_AND_MATCH_CPU_WORKERS = 36

###############################################################################
# ALIGN_COMPUTE_TRANSFORMS params
###############################################################################

###############################################################################
# ALIGN_COMPUTE_WARPS params
###############################################################################

N_ALIGN_COMPUTE_WARPS_WORKERS_PER_CPU = 1
N_ALIGN_COMPUTE_WARPS_CPU_WORKERS = 36

###############################################################################
# PREPARE DATA General params
###############################################################################
IS_3D_FIX = 0

CNN_PATCH_LEG = 26
CNN_PATCH_LEG = 26

###############################################################################
# PREPARE DATA CC params
###############################################################################

PREPARE_DATA_CC_PATH = '/home/armafire/Pipeline/matlab_scripts/run_cc_block.sh'

N_PREPARE_DATA_CC_WORKERS_PER_CPU = 8 * 2

###############################################################################
# PREPARE DATA BLACKEN params
###############################################################################

PREPARE_DATA_BLACKEN_PATH = '/home/armafire/Pipeline/matlab_scripts/run_blacken.sh'

N_PREPARE_DATA_BLACKEN_WORKERS_PER_CPU = 8 * 2

###############################################################################
# PREPARE DATA SUBBLOCKS params
###############################################################################

PREPARE_DATA_SUBBLOCKS_IS_IGNORE = 1

PREPARE_DATA_SUBBLOCKS_PATH = 'python /home/armafire/Pipeline/datasets/scripts/block_to_segments.py'

N_PREPARE_DATA_SUBBLOCKS_WORKERS_PER_CPU = 8 * 2

###############################################################################
# BORDER MASKS params
###############################################################################

BORDER_WIDTH = 5
CLOSE_WIDTH = 26

N_BORDER_MASKS_WORKERS_PER_CPU = 8 * 2

BORDER_MASKS_PATH = '/home/armafire/Pipeline/matlab_scripts/run_border_masks.sh'

###############################################################################
# COMBINE BORDER MASKS params
###############################################################################

N_COMBINE_BORDER_MASKS_WORKERS_PER_CPU = 8 * 2

COMBINE_BORDER_MASKS_PATH = 'python /home/armafire/Pipeline/datasets/scripts/combine_masks.py'

###############################################################################
# CNN params
###############################################################################

N_FC_DNN_WORKERS_PER_CPU = 1
N_PER_FC_DNN_WORKERS = 36

FC_DNN_PATH = '/home/armafire/Pipeline/fc_dnn/fully_conv/fully_conv/run_dnn_53_dist_4_3D_PAD'

FC_DNN_3D_DEPTH = 3

FC_DNN_N_CHANNELS = 4

###############################################################################
# PROBS_COMBINE params
###############################################################################

N_PROBS_COMBINE_WORKERS_PER_CPU = 8 * 2

PROBS_COMBINE_PATH = '/home/armafire/Pipeline/matlab_scripts/run_extractSeeds.sh'

PROBS_COMBINE_KERNEL_SIZE = 13
PROBS_COMBINE_REGIONAL_DEPTH_REDUCE = 0.04
PROBS_COMBINE_STD_SMOOTH = 2.0

###############################################################################
# PREPARE WS_NP_PREPARE params
###############################################################################

IS_Z_DEPTH_CUT = 0

BLOCK_OUTER_MASK_FILENAME = 't_outer_mask_all.png'
BLOCK_BORDER_MASK_FILENAME = 't_border_mask_all.png'

OUTER_MASK_FILENAME = 'blocks_%.4d_%.4d_outer_mask_total.png'
BORDER_MASK_FILENAME = 'blocks_%.4d_%.4d_border_mask_total.png'
ALL_BLOCKS_META = 'all_blocks_%.4d_%.4d_meta'

WS_NP_PREPARE_PATH = '/home/armafire/Pipeline/matlab_scripts/run_ws_np_prepare.sh'

N_WS_NP_PREPARE_WORKERS_PER_CPU = 8 * 2

###############################################################################
# WS params
###############################################################################

N_WS_WORKERS_PER_CPU = 8 * 2

WS_PATH = '/home/armafire/Pipeline/exec/watershed/cpp/build_pipeline/watershed-3D-BG.x'

###############################################################################
# NP params
###############################################################################

NP_INPUTS_IS_OVERWRITE = 1

N_NP_INPUT_PREPARE_WORKERS_PER_CPU = 8 * 2

N_NP_EXEC_WORKERS_PER_CPU = 4 * 2

N_PER_NP_EXEC_CILK_WORKERS = 4

NP_PATH = '/home/armafire/Pipeline/exec/neuroproof_agg/npclean/build/neuroproof_graph_predict'

NP_CLASSIFIER_DIR = '/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-K11-AC-256-3D-PAD/'
#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-K11-S1-AC3-256-53-dist-4-GT/'

#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-K11-AC-256-3D-PAD-FM25/'
#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-P3-errors-3D-49x49-32f-GT/' 
#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-K11-AC-256-3D-PAD-FM25/'
#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-P3-errors-NEW-2-3D-49x49-32f-GT-ws-2D'
#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-P3-errors-NEW-2-3D-49x49-32f-GT-ws-2D'
#'/home/armafire/Pipeline/exec_full/np_classifiers/np-classifier-P3-errors-NEW-2-3D-49x49-32f-GT/'

###############################################################################
# BLOCK_VIDEO params
###############################################################################

BLOCK_VIDEO_PATH = '/home/armafire/Pipeline/matlab_scripts/run_mergeH5_1core.sh'

N_BLOCK_VIDEO_WORKERS_PER_CPU = 8

IS_BLOCK_VIDEO = 1

###############################################################################
# MERGE_PREPROCESS params
###############################################################################

MERGE_BASE_DIR = '/home/armafire/Pipeline/np_merge_v2'

MERGE_PREPROCESS_PATH = os.path.join(MERGE_BASE_DIR, 'np-merge/python/run_preprocess.sh')

N_MERGE_PREPROCESS_WORKERS_PER_CPU = 8 * 2

###############################################################################
# MERGE_EXEC params
###############################################################################

MERGE_EXEC_PATH = os.path.join(MERGE_BASE_DIR, 'np-merge/python/run_block_merge.sh')

N_MERGE_EXEC_WORKERS_PER_CPU = 8 * 2

N_MERGE_EXEC_CILK_WORKERS_PER_RUN = 4

###############################################################################
# MERGE_COMBINE params
###############################################################################

MERGE_COMBINE_PATH = os.path.join(MERGE_BASE_DIR, 'np-merge/cpp/run_combine.sh')

N_MERGE_COMBINE_WORKERS_PER_CPU = 8

###############################################################################
# MERGE_RELABEL params
###############################################################################

MERGE_END_BLOCK_DEPTH = 20
MERGE_END_BLOCK_ROW = 20
MERGE_END_BLOCK_COL = 20

MERGE_RELABEL_PATH = os.path.join(MERGE_BASE_DIR, 'np-merge/cpp/run_relabel.sh')

N_MERGE_RELABEL_WORKERS_PER_CPU = 32

###############################################################################
# SKELETON params
###############################################################################

N_SKELETON_WORKERS_PER_CPU = 8

N_PER_SKELETON_EXEC_CILK_WORKERS = 4

SKELETON_PATH = '/home/armafire/Pipeline/skeletons/mit3dsegmentation/graph_extraction/run_skeletonization.sh'

###############################################################################
# DEBUG_GENERATE params
###############################################################################

DEBUG_OUT_PREFIX = 'K11_S1'

DEBUG_GENERATE_PATH = 'python debug_generate.py'

N_DEBUG_GENERATE_WORKERS_PER_CPU = 18

###############################################################################
###############################################################################

def verify_and_get_np_classifier(np_classifier_dir):
    print 'verify_and_get_np_classifier:'
    print ' -- np_classifier_dir: %s' % (np_classifier_dir,)
    
    filenames = os.listdir(np_classifier_dir)
    
    if len(filenames) != 2:
        raise Exception('ERROR: Only 2 files are expected for neuroproof classifier (xml and txt) [got: %d]' %
                        (len(filenames),))
    
    filename_1 = filenames[0]
    filename_2 = filenames[1]
    
    if filename_1.find('_ignore.txt') != -1:
        np_txt_filename = filename_1
    elif filename_2.find('ignore.txt') != -1:
        np_txt_filename = filename_2
    else:
        raise Exception('ERROR: *_ignore.txt neuroproof file was not found')

    if filename_1.find('.xml') != -1:
        np_xml_filename = filename_1
    elif filename_2.find('.xml') != -1:
        np_xml_filename = filename_2
    else:
        raise Exception('ERROR: *.xml neuroproof file was not found')
    
    np_xml_filepath = os.path.join(np_classifier_dir, np_xml_filename)
    
    print ' -- np_classifier_file: %s' % (np_xml_filename,)
    return np_xml_filepath
    
def get_numa_cmd(cpu_id):
    
    return 'numactl --cpunodebind=%d' % (cpu_id,)

def get_per_core_numa_cmd(job_id):

    return 'numactl --physcpubind=%d' % (job_id)
    
def exec_process(cmd, log_filepath, cmd_env = None):
    print 'exec_process: '
    print ' CMD: %s' % (cmd,)
    print ' LOG: %s' % (log_filepath,)
    
    if IS_FLUSH_LOGS:
        f_log = open(log_filepath, 'wb', 0)
    else:
        f_log = open(log_filepath, 'wb')
    
    proc = subprocess.Popen(
        cmd,
        shell = True,
        stdin = None,
        stdout = f_log,
        stderr = f_log,
        env = cmd_env)
    
    return proc 

def jobs_sync(max_procs, cur_procs, procs_data, is_all):
    
    n_procs = len(cur_procs)
        
    print ' -- jobs_sync start: %d PROCESSES RUNNING [IS_ALL = %d]' % (n_procs, is_all)
    
    while (True):
        
        if not is_all:
            if (len(cur_procs) < max_procs):
                break
        else:
            if (len(cur_procs) == 0):
                break
        
        time.sleep(0.5)
        
        for i, proc in enumerate(cur_procs):
            if proc.poll() != None:
                cur_procs.remove(proc)
                
                (cmd, job_id, start_time) = procs_data[i]
                procs_data.remove((cmd, job_id, start_time))
                
                elapsed_time = time.time() - start_time
                    
                print ' -- jobs_sync: PROCESS %d DONE' % (job_id,)
                print '    -- CMD: %s' % (cmd,)
                print '    -- TIME: %d [secs]' % (elapsed_time,)
    
    print ' -- jobs_sync finish: %d PROCESSES RUNNING' % (len(cur_procs),)                
    return
    

def jobs_exec_process(max_procs, cur_procs, procs_data, job_id, cmd, log_filepath, cmd_env = None):
    
    jobs_sync(max_procs, cur_procs, procs_data, False)
    
    start_time = time.time()
    
    proc = exec_process(cmd, log_filepath, cmd_env)
    
    cur_procs.append(proc)
    
    procs_data.append((cmd, job_id, start_time))
    
    print 'jobs_exec_process: %d' % (len(cur_procs),)
    
    return proc

def get_segments(block_path):
    filenames = os.listdir(block_path)
    
    seg_names = []
    
    for filename in filenames:
        if not os.path.isdir(os.path.join(block_path, filename)):
            continue
        if filename.find('_segment_') == -1:
            continue
        if filename.find('_type_complex') != -1:
            continue
        
        seg_names.append(filename)
    
    seg_names.sort()
    
    res_segments = []
    for seg_name in seg_names:
        
        parts = seg_name.split('_range_')[1].split('_type_')[0].split('_')
        start_id = int(parts[0])
        finish_id = int(parts[1])
        
        n_seg_files = finish_id - start_id + 1; 
        
        res_segments.append((seg_name, n_seg_files))
    
    return res_segments

def is_block_exists(block_depth, block_row_id, block_col_id):
    block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
    
    block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
    
    if (len(os.listdir(block_path)) == 0):
        print 'SKIP %s: no files' % (block_name,)
        return False
    
    return True
    

def is_block_valid(block_depth, block_row_id, block_col_id):
    
    if not is_block_exists(block_depth, block_row_id, block_col_id):
        return False
    
    block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
    
    block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
    block_path_meta = os.path.join(BLOCKS_DIR_META, '%s_meta' % (block_name,))
    
    if (len(os.listdir(block_path_meta)) == 0):
        print 'SKIP %s: no meta files' % (block_name,)
        return False
    
    segments = get_segments(block_path)
    
    for (seg_name, n_seg_files) in segments:
        seg_path = os.path.join(block_path, seg_name)
        seg_path_meta = os.path.join(block_path_meta, '%s_meta' % (seg_name,))
        
        if (len(os.listdir(seg_path_meta)) == 0):
            print 'SKIP %s: no meta files' % (block_name,)
            return False
        
        seg_outer_mask_path = os.path.join(seg_path_meta, 't_outer_mask_all.png')
        outer_mask = cv2.imread(seg_outer_mask_path, cv2.IMREAD_UNCHANGED)
        
        assert (outer_mask.dtype == np.uint8)
        
        d1 = outer_mask.shape[0]
        d2 = outer_mask.shape[1]
        
        average_val = (float(outer_mask.sum()) / float(d1 * d2))
        if (average_val > 220.0):
            print 'SKIP %s: excessive mask (%f)' % (block_name, average_val)
            return False
        
    return True
    
#####
def ALIGN_GENERATE_TILES_execute():

    print 'ALIGN_GENERATE_TILES_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_ALIGN_GENERATE_TILES_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    cur_section = N_ALIGN_BASE_SECTION
    finish_section = N_ALIGN_BASE_SECTION + N_ALIGN_SECTIONS
    
    if not os.path.exists(BLOCKS_DIR_ALIGN_TILES):
        os.makedirs(BLOCKS_DIR_ALIGN_TILES)
    
    while cur_section < finish_section:
        
        base_section = cur_section
        n_sections = N_ALIGN_BATCH
        cur_section += N_ALIGN_BATCH
        
        if cur_section > finish_section:
            cur_section = finish_section
        
        print ' -- ALIGN_GENERATE_TILES: [%d -> %d] ' % (base_section, base_section + n_sections)
        
        align_generate_tiles_cmd = ('%s %s %d %d %s' % 
            (ALIGN_TILES_PATH,
             ALIGN_EM_SOURCE,
             base_section,
             n_sections,
             BLOCKS_DIR_ALIGN_TILES))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, align_generate_tiles_cmd)
        
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, 'align_generate_tiles_%.4d_%.4d.log' % (base_section, n_sections))
        cmd_env = os.environ.copy()
        #cmd_env['CILK_NWORKERS'] = '%d' % (N_ALIGN_CPU_WORKERS,)
        
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ALIGN_GENERATE_TILES_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        

###
def ALIGN_GENERATE_TILES_TXT_execute():

    print 'ALIGN_GENERATE_TILES_TXT_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = 1
    n_jobs = 0
    procs = []
    procs_data = []
    
    if not os.path.exists(BLOCKS_DIR_ALIGN_TILES_TXT):
        os.makedirs(BLOCKS_DIR_ALIGN_TILES_TXT)
    
    align_generate_tiles_txt_cmd = ('%s %s %d %d %s' % 
        (ALIGN_TILES_TXT_PATH,
         BLOCKS_DIR_ALIGN_TILES,
         N_ALIGN_BASE_SECTION,
         N_ALIGN_SECTIONS,
         TILES_TXT_FILEPATH))
    
    numactl_cmd = get_numa_cmd(0)
    
    final_cmd = '%s %s' % (numactl_cmd, align_generate_tiles_txt_cmd)
    
    log_filepath = os.path.join(BLOCKS_DIR_LOGS, 'align_generate_tiles_txt_%.4d_%.4d.log' % 
        (N_ALIGN_BASE_SECTION, N_ALIGN_BASE_SECTION + N_ALIGN_SECTIONS))
    cmd_env = os.environ.copy()
    
    jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
    
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ALIGN_GENERATE_TILES_TXT_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        
    

def ALIGN_COMPUTE_KPS_AND_MATCH_execute():

    print 'ALIGN_COMPUTE_KPS_AND_MATCH_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_ALIGN_COMPUTE_KPS_AND_MATCH_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    cur_section = N_ALIGN_BASE_SECTION
    
    if not os.path.exists(BLOCKS_DIR_ALIGN_WORK_DIR):
        os.makedirs(BLOCKS_DIR_ALIGN_WORK_DIR)
        
    if not os.path.exists(BLOCKS_DIR_ALIGN_RESULT_DIR):
        os.makedirs(BLOCKS_DIR_ALIGN_RESULT_DIR)
    
    while cur_section < (N_ALIGN_BASE_SECTION + N_ALIGN_SECTIONS):
        
        base_section = cur_section
        n_sections = N_ALIGN_BATCH
        cur_section += (N_ALIGN_BATCH - 1)
        
        print ' -- ALIGN_COMPUTE_KPS_AND_MATCH: [%d -> %d] ' % (base_section, base_section + n_sections)
        
        if ((base_section + n_sections) > (N_ALIGN_BASE_SECTION + N_ALIGN_SECTIONS)):
            n_sections = (N_ALIGN_BASE_SECTION + N_ALIGN_SECTIONS) - base_section
        
        align_compute_kps_and_match_cmd = ('%s 1 %d %d %s %s %s' % 
            (ALIGN_PATH,
             base_section,
             n_sections,
             TILES_TXT_FILEPATH,
             BLOCKS_DIR_ALIGN_WORK_DIR,
             BLOCKS_DIR_ALIGN_RESULT_DIR))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, align_compute_kps_and_match_cmd)
        
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, 'align_compute_kps_and_match_%.4d_%.4d.log' % (base_section, n_sections))
        cmd_env = os.environ.copy()
        cmd_env['CILK_NWORKERS'] = '%d' % (N_ALIGN_COMPUTE_KPS_AND_MATCH_CPU_WORKERS,)
        
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ALIGN_COMPUTE_KPS_AND_MATCH_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        

def ALIGN_COMPUTE_TRANSFORMS_execute():

    print 'ALIGN_COMPUTE_TRANSFORMS_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = 1
    n_jobs = 0
    procs = []
    procs_data = []
    cur_section = N_ALIGN_BASE_SECTION
    
    print ' -- ALIGN_COMPUTE_KPS_AND_MATCH: [%d -> %d] ' % (
        N_ALIGN_BASE_SECTION, 
        N_ALIGN_BASE_SECTION + N_ALIGN_SECTIONS)
    
    align_compute_transforms_cmd = ('%s 2 %d %d %s %s %s' % 
        (ALIGN_PATH,
         N_ALIGN_BASE_SECTION,
         N_ALIGN_SECTIONS,
         TILES_TXT_FILEPATH,
         BLOCKS_DIR_ALIGN_WORK_DIR,
         BLOCKS_DIR_ALIGN_RESULT_DIR))
    
    numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
    
    final_cmd = '%s %s' % (numactl_cmd, align_compute_transforms_cmd)
    
    log_filepath = os.path.join(BLOCKS_DIR_LOGS, 'align_compute_transforms_%.4d_%.4d.log' % 
        (N_ALIGN_BASE_SECTION, N_ALIGN_BASE_SECTION + N_ALIGN_SECTIONS))
    cmd_env = os.environ.copy()
    
    jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
    
    n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ALIGN_COMPUTE_TRANSFORMS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        

##
def ALIGN_COMPUTE_WARPS_execute():

    print 'ALIGN_COMPUTE_WARPS_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_ALIGN_COMPUTE_WARPS_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    cur_section = N_ALIGN_BASE_SECTION
    
    while cur_section < (N_ALIGN_BASE_SECTION + N_ALIGN_SECTIONS):
        
        base_section = cur_section
        n_sections = N_ALIGN_BATCH
        cur_section += N_ALIGN_BATCH
        
        print ' -- ALIGN_COMPUTE_WARPS: [%d -> %d] ' % (base_section, base_section + n_sections)
        
        align_compute_warps_cmd = ('%s 3 %d %d %s %s %s' % 
            (ALIGN_PATH,
             base_section,
             n_sections,
             TILES_TXT_FILEPATH,
             BLOCKS_DIR_ALIGN_WORK_DIR,
             BLOCKS_DIR_ALIGN_RESULT_DIR))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, align_compute_warps_cmd)
        
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, 'align_compute_warps_%.4d_%.4d.log' % (base_section, n_sections))
        cmd_env = os.environ.copy()
        cmd_env['CILK_NWORKERS'] = '%d' % (N_ALIGN_COMPUTE_WARPS_CPU_WORKERS,)
        
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ALIGN_COMPUTE_WARPS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        

#####
def NP_execute_prepare(blocks):
    
    print 'NP_execute_prepare: start'
    
    start_time_secs = time.time()
        
    max_jobs = N_NP_INPUT_PREPARE_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        np_block_n_depth = BLOCK_N_DEPTH
        if IS_3D_FIX:
            if block_depth == 0:
                np_block_n_depth -= 2
        
        np_block_n_rows = BLOCK_N_ROWS
        np_block_n_cols = BLOCK_N_COLS
        if block_row_id == 0:
            np_block_n_rows -= CNN_PATCH_LEG
        if block_row_id == BLOCKS_MAX_ROW_ID:
            np_block_n_rows -= CNN_PATCH_LEG
        if block_col_id == 0:
            np_block_n_cols -= CNN_PATCH_LEG
        if block_col_id == BLOCKS_MAX_COL_ID:
            np_block_n_cols -= CNN_PATCH_LEG
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
        if IS_WS_NP_PREPARE:
            block_path_probs = os.path.join(BLOCKS_DIR_PROBS_NP, '%s_probs' % (block_name,))
        else:
            block_path_probs = os.path.join(BLOCKS_DIR_PROBS, '%s_probs' % (block_name,))
            
        block_path_ws = os.path.join(BLOCKS_DIR_WS, '%s_ws' % (block_name,))
        block_path_np = os.path.join(BLOCKS_DIR_NP, '%s_np' % (block_name,))        
        
        if not is_block_valid(block_depth, block_row_id, block_col_id):
            continue
        
        segments = get_segments(block_path)
        
        for (seg_name, n_seg_files) in segments:
            seg_path_probs = os.path.join(block_path_probs, '%s_probs' % (seg_name,))
            seg_path_ws = os.path.join(block_path_ws, '%s_ws' % (seg_name,))
            seg_path_np = os.path.join(block_path_np, '%s_np' % (seg_name,))
            
            is_overwrite = 0
            if not os.path.exists(seg_path_np):
                os.makedirs(seg_path_np)
            else:
                is_overwrite = 1
            
            print ' -- %s %s [%d,%d,%d] [overwrite = %d]' % (block_name, seg_name, n_seg_files, np_block_n_rows, np_block_n_cols, is_overwrite)
            print ' -- INPUT PROBS : %s' % (seg_path_probs,)
            print ' -- INPUT WS    : %s' % (seg_path_ws,)
            print ' -- OUTPUT      : %s' % (seg_path_np,)
            
            np_generate_inputs_cmd = ('python np_generate_inputs.py %d %s %s %s %s' %
                (NP_INPUTS_IS_OVERWRITE,
                 seg_path_probs,
                 seg_path_ws,
                 seg_path_np,
                 seg_name))
            
            numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
            
            final_cmd = '%s %s' % (numactl_cmd, np_generate_inputs_cmd)
            
            log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_np_prepare.log' % (seg_name,))
            cmd_env = os.environ.copy()
            #cmd_env['CILK_NWORKERS'] = '%d' % (WORKERS_PER_CPU,)
            
            jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
            
            n_jobs += 1
            
    
    jobs_sync(max_jobs, procs, procs_data, True)
        
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'NP_execute_prepare: finish [total_time = %d secs]' % (elapsed_time_secs,)
        

def NP_execute(blocks):
    
    print 'NP_execute: start'
    
    start_time_secs = time.time()
    
    np_classifier_filepath = verify_and_get_np_classifier(NP_CLASSIFIER_DIR)
    
    max_jobs = N_NP_EXEC_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        np_block_n_depth = BLOCK_N_DEPTH
        if IS_3D_FIX:
            if block_depth == 0:
                np_block_n_depth -= 2
        
        np_block_n_rows = BLOCK_N_ROWS
        np_block_n_cols = BLOCK_N_COLS
        if block_row_id == 0:
            np_block_n_rows -= CNN_PATCH_LEG
        if block_row_id == BLOCKS_MAX_ROW_ID:
            np_block_n_rows -= CNN_PATCH_LEG
        if block_col_id == 0:
            np_block_n_cols -= CNN_PATCH_LEG
        if block_col_id == BLOCKS_MAX_COL_ID:
            np_block_n_cols -= CNN_PATCH_LEG
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
        if IS_WS_NP_PREPARE:
            block_path_probs = os.path.join(BLOCKS_DIR_PROBS_NP, '%s_probs' % (block_name,))
        else:
            block_path_probs = os.path.join(BLOCKS_DIR_PROBS, '%s_probs' % (block_name,))
        block_path_ws = os.path.join(BLOCKS_DIR_WS, '%s_ws' % (block_name,))
        block_path_np = os.path.join(BLOCKS_DIR_NP, '%s_np' % (block_name,))        
        
        if not is_block_valid(block_depth, block_row_id, block_col_id):
            continue
        
        segments = get_segments(block_path)
        
        for (seg_name, n_seg_files) in segments:
            seg_path_probs = os.path.join(block_path_probs, '%s_probs' % (seg_name,))
            seg_path_ws = os.path.join(block_path_ws, '%s_ws' % (seg_name,))
            seg_path_np = os.path.join(block_path_np, '%s_np' % (seg_name,))
            
            is_overwrite = 0
            if not os.path.exists(seg_path_np):
                raise Exception('NP segment directory is missing: %s' % (seg_path_np,))
            else:
                is_overwrite = 1
            
            print ' -- %s %s [%d,%d,%d] [overwrite = %d]' % (block_name, seg_name, n_seg_files, np_block_n_rows, np_block_n_cols, is_overwrite)
            print ' -- INPUT NP : %s' % (seg_path_np,)
            print ' -- OUTPUT   : %s' % (seg_path_np,)
            
            np_exec_cmd = ('python np_predict.py %s %s %s %s' % 
                (NP_PATH,
                 np_classifier_filepath,
                 seg_path_np,
                 seg_name))
                 
            numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
            
            final_cmd = '%s %s' % (numactl_cmd, np_exec_cmd)
            
            log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_np_exec.log' % (seg_name,))
            cmd_env = os.environ.copy()
            cmd_env['CILK_NWORKERS'] = '%d' % (N_PER_NP_EXEC_CILK_WORKERS,)
            
            jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
            
            n_jobs += 1
            
            
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'NP_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)
    
def block_video(blocks):
    
    print 'block_video: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_BLOCK_VIDEO_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        np_block_n_depth = BLOCK_N_DEPTH
        if IS_3D_FIX:
            if block_depth == 0:
                np_block_n_depth -= 2
        
        np_block_n_rows = BLOCK_N_ROWS
        np_block_n_cols = BLOCK_N_COLS
        if block_row_id == 0:
            np_block_n_rows -= CNN_PATCH_LEG
        if block_row_id == BLOCKS_MAX_ROW_ID:
            np_block_n_rows -= CNN_PATCH_LEG
        if block_col_id == 0:
            np_block_n_cols -= CNN_PATCH_LEG
        if block_col_id == BLOCKS_MAX_COL_ID:
            np_block_n_cols -= CNN_PATCH_LEG
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
        
        block_path_np = os.path.join(BLOCKS_DIR_NP, '%s_np' % (block_name,))        
        
        if not is_block_valid(block_depth, block_row_id, block_col_id):
            continue
        
        print ' -- %s [%d,%d,%d]' % (block_name, np_block_n_depth, np_block_n_rows, np_block_n_cols)
        print ' -- INPUT NP : %s' % (block_path_np,)
        print ' -- OUTPUT   : %s' % (block_path_np,)
        
        block_video = ('%s %s %s* %d %d %d %d %s' % 
            (BLOCK_VIDEO_PATH,
             BLOCKS_DIR_NP,
             block_name,
             np_block_n_rows,
             np_block_n_cols,
             np_block_n_depth,
             IS_BLOCK_VIDEO,
             BLOCKS_DIR_EM_CC))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, block_video)
        
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_block_video.log' % (block_name,))
        cmd_env = os.environ.copy()
        
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1
        
    
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'block_video: finish [total_time = %d secs]' % (elapsed_time_secs,)

def MERGE_preprocess(blocks):
    
    print 'MERGE_preprocess: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_MERGE_PREPROCESS_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
        
        if not is_block_valid(block_depth, block_row_id, block_col_id):
            continue
                
        merge_preprocess = ('%s %d %d %d %d %s %s %s %s' % 
            (MERGE_PREPROCESS_PATH,
             block_depth,
             block_row_id,
             block_col_id,
             BLOCK_N_DEPTH,
             BLOCKS_DIR_MERGE,
             BLOCKS_DIR_PROBS_WS,
             BLOCKS_DIR_PROBS_NP,
             BLOCKS_DIR_NP))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, merge_preprocess)
        
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_merge_preprocess.log' % (block_name,))
        cmd_env = os.environ.copy()
        
        proc = jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'MERGE_preprocess: finish [total_time = %d secs]' % (elapsed_time_secs,)
    

def MERGE_exec(blocks):
    
    print 'MERGE_exec: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_MERGE_EXEC_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
        
        if not is_block_valid(block_depth, block_row_id, block_col_id):
            continue
        
        merge_exec = ('%s %s %d %d %d' % 
            (MERGE_EXEC_PATH,
             BLOCKS_DIR_MERGE,
             block_depth,
             block_row_id,
             block_col_id))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, merge_exec)
        
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_merge_exec.log' % (block_name,))
        cmd_env = os.environ.copy()
        cmd_env = os.environ.copy()
        cmd_env['CILK_NWORKERS'] = '%d' % (N_MERGE_EXEC_CILK_WORKERS_PER_RUN,)
        
        proc = jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'MERGE_exec: finish [total_time = %d secs]' % (elapsed_time_secs,)

# def merge_sync_procs(procs_to_blocks, blocks_running):
#     
#     for proc in procs_to_blocks.keys():
#         if proc.poll() != None:
#             blocks = procs_to_blocks[proc]
#             print '-- MERGE: process %r finished' % (proc,)
#             print '     blocks: %r' % (blocks,)
#             
#             for block in blocks:
#                 blocks_running.remove(block)
#             
#             procs_to_blocks.pop(proc)
# 
# def MERGE_exec(blocks):
#     
#     print 'MERGE_exec: start'
#     
#     start_time_secs = time.time()
#     
#     max_jobs = N_MERGE_EXEC_WORKERS_PER_CPU * N_CPUS
#     n_jobs = 0
#     procs = []
#     procs_data = []
#     
#     blocks_running = []
#     procs_to_blocks = {}
#     blocks_to_process = blocks[:]
#     blocks_to_process.sort()
#     
#     while (len(blocks_to_process) > 0): 
#         block = blocks_to_process[0]
#         
#         if block in blocks_running:
#             print '-- Block %r is running' % (block,)
#             print '-- Searching for alternative block'
#             
#             while (True):
#                 merge_sync_procs(procs_to_blocks, blocks_running)
#                 
#                 is_found = 0
#             
#                 for i in xrange(len(blocks_to_process)):
#                     block = blocks_to_process[i]
#                 
#                     if block not in blocks_running:
#                         adj_block_Z = (block[0] + 1, block[1], block[2])
#                         adj_block_row = (block[0], block[1] + 1, block[2])
#                         adj_block_col = (block[0], block[1], block[2] + 1)
#                         
#                         if ((adj_block_Z not in blocks_running) and 
#                             (adj_block_row not in blocks_running) and 
#                             (adj_block_col not in blocks_running)):    
#                             is_found = 1
#                             break
#             
#                 if not is_found:
#                     print '-- WAITING FOR SOME PROCESS TO FINISH'
#                     jobs_sync(len(procs), procs, procs_data, False)
#                 else:
#                     print '-- Found block %r' % (block,)
#                     break
#         
#         block_depth = block[0]
#         block_row_id = block[1]
#         block_col_id = block[2]
#                 
#         block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
#         block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
#         
#         block_path_np = os.path.join(BLOCKS_DIR_NP, '%s_np' % (block_name,))        
#         
#         if (len(os.listdir(block_path)) == 0):
#             blocks_to_process.remove(block)
#             print 'SKIP %s: no files' % (block_name,)
#             continue
#         
#         print ' -- %s ' % (block_name,)
#         print ' -- INPUT NP : %s' % (block_path_np,)
#         print ' -- OUTPUT   : %s' % (block_path_np,)
#         
#         merge_exec = ('%s %s %s %s %d %d %d' % 
#             (MERGE_EXEC_PATH,
#              BLOCKS_DIR_NP,
#              BLOCKS_DIR_META,
#              BLOCKS_DIR_MERGE,
#              block_depth,
#              block_row_id,
#              block_col_id))
#              
#         numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
#         
#         final_cmd = '%s %s' % (numactl_cmd, merge_exec)
#         
#         log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_merge_exec.log' % (block_name,))
#         cmd_env = os.environ.copy()
#         cmd_env['CILK_NWORKERS'] = '%d' % (MERGE_EXEC_CILK_WORKERS_PER_RUN,)
#         
#         proc = jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
#         
#         n_jobs += 1
#         
#         proc_blocks = []
#         proc_blocks.append((block_depth, block_row_id, block_col_id))
#         proc_blocks.append((block_depth + 1, block_row_id, block_col_id))
#         proc_blocks.append((block_depth, block_row_id + 1, block_col_id))
#         proc_blocks.append((block_depth, block_row_id, block_col_id + 1))
# 
#         procs_to_blocks[proc] = proc_blocks
#         
#         blocks_running += proc_blocks
#         
#         blocks_to_process.remove(block)
#         
#         merge_sync_procs(procs_to_blocks, blocks_running)
#         
#     
#     jobs_sync(max_jobs, procs, procs_data, True)
#     
#     elapsed_time_secs = time.time() - start_time_secs
#     
#     print 'MERGE_exec: finish [total_time = %d secs]' % (elapsed_time_secs,)
#     

def MERGE_combine(blocks):
    
    print 'MERGE_combine: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_MERGE_COMBINE_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []

    merge_combine = ('%s %s %d %d %d %d %d %d' % 
        (MERGE_COMBINE_PATH,
         BLOCKS_DIR_MERGE,
         blocks[0][0],
         blocks[0][1],
         blocks[0][2],
         blocks[-1][0]+1,
         blocks[-1][1]+1,
         blocks[-1][2]+1,))
    
    numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
    
    final_cmd = '%s %s' % (numactl_cmd, merge_combine)
    
    log_filepath = os.path.join(BLOCKS_DIR_LOGS, 'all_blocks_merge_combine.log')
    cmd_env = os.environ.copy()
    
    proc = jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
    
    n_jobs += 1
    
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'MERGE_combine: finish [total_time = %d secs]' % (elapsed_time_secs,)


def MERGE_relabel(blocks):
    
    print 'MERGE_relabel: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_MERGE_RELABEL_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
        
        if not is_block_valid(block_depth, block_row_id, block_col_id):
            continue
        
        merge_relabel = ('%s %s %d %d %d %d %d %d' % 
            (MERGE_RELABEL_PATH,
             BLOCKS_DIR_MERGE,
             block_depth,
             block_row_id,
             block_col_id,
             MERGE_END_BLOCK_DEPTH,
             MERGE_END_BLOCK_ROW,
             MERGE_END_BLOCK_COL))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, merge_relabel)
        
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_merge_relabel.log' % (block_name,))
        cmd_env = os.environ.copy()
        
        proc = jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'MERGE_relabel: finish [total_time = %d secs]' % (elapsed_time_secs,)
    

def skeleton_exec(blocks):
    
    print 'skeleton_blocks: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_SKELETON_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []

    for block in blocks:
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
        block_path_merge = BLOCKS_DIR_MERGE
        block_path_skeleton = os.path.join(BLOCKS_DIR_SKELETONS, '%s_skeleton' % (block_name,))        
        
        if not is_block_valid(block_depth, block_row_id, block_col_id):
            continue
        
        block_path_merge_h5_file = os.path.join(block_path_merge, 'out_segmentation_%.4d_%.4d_%.4d.h5' % (block_depth,
                                                                                                                block_row_id,
                                                                                                                block_col_id))
        block_path_skeleton_SWC = os.path.join(block_path_skeleton, 'SWC')
            
        is_overwrite = 0
        if not os.path.exists(block_path_skeleton_SWC):
            os.makedirs(block_path_skeleton_SWC)
        else:
            is_overwrite = 1
                
        print ' -- %s [overwrite = %d]' % (block_name, is_overwrite)
        print ' -- INPUT NP : %s' % (block_path_merge_h5_file,)
        print ' -- OUTPUT   : %s' % (block_path_skeleton,)
            
        skeleton_cmd = ('%s %s %s' % 
            (SKELETON_PATH,
             block_path_merge_h5_file,
             block_path_skeleton))
                 
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
            
        final_cmd = '%s %s' % (numactl_cmd, skeleton_cmd)
            
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_skeleton_exec.log' % (block_name,))
        cmd_env = os.environ.copy()
        cmd_env['CILK_NWORKERS'] = '%d' % (N_PER_SKELETON_EXEC_CILK_WORKERS,)
            
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
            
        n_jobs += 1
    
    jobs_sync(max_jobs, procs, procs_data, True)
        
    elapsed_time_secs = time.time() - start_time_secs
        
    print 'skeleton_exec: finish [total_time = %d secs]' % (elapsed_time_secs,)

def WS_execute(blocks):
    
    print 'WS_execute: start'

    start_time_secs = time.time()
        
    max_jobs = N_WS_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        ws_block_n_depth = BLOCK_N_DEPTH
        if IS_3D_FIX:
            if block_depth == 0:
                ws_block_n_depth -= 2
        
        ws_block_n_rows = BLOCK_N_ROWS
        ws_block_n_cols = BLOCK_N_COLS
        if block_row_id == 0:
            ws_block_n_rows -= CNN_PATCH_LEG
        if block_row_id == BLOCKS_MAX_ROW_ID:
            ws_block_n_rows -= CNN_PATCH_LEG
        if block_col_id == 0:
            ws_block_n_cols -= CNN_PATCH_LEG
        if block_col_id == BLOCKS_MAX_COL_ID:
            ws_block_n_cols -= CNN_PATCH_LEG
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)

        if IS_WS_NP_PREPARE:
            block_path_probs = os.path.join(BLOCKS_DIR_PROBS_WS, '%s_probs' % (block_name,))
        else:
            block_path_probs = os.path.join(BLOCKS_DIR_PROBS, '%s_probs' % (block_name,))
        
        block_path_ws = os.path.join(BLOCKS_DIR_WS, '%s_ws' % (block_name,))        
        
        if not is_block_valid(block_depth, block_row_id, block_col_id):
            continue
        
        segments = get_segments(block_path)
        
        for (seg_name, n_seg_files) in segments:
            seg_path_probs = os.path.join(block_path_probs, '%s_probs' % (seg_name,))
            seg_path_ws = os.path.join(block_path_ws, '%s_ws' % (seg_name,))
        

            is_overwrite = 0
            if not os.path.exists(seg_path_ws):
                os.makedirs(seg_path_ws)
            else:
                is_overwrite = 1

            print ' -- %s %s [%d,%d,%d] [overwrite = %d]' % (block_name, seg_name, n_seg_files, ws_block_n_rows, ws_block_n_cols, is_overwrite)
            print ' -- INPUT  : %s' % (seg_path_probs,)
            print ' -- OUTPUT : %s' % (seg_path_ws,)
        
            ws_cmd = ('%s %s %s' % 
                (WS_PATH,
                seg_path_probs,
                seg_path_ws))
        
            numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
            final_cmd = '%s %s' % (numactl_cmd, ws_cmd)

            log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_ws.log' % (seg_name,))
            cmd_env = os.environ.copy()
            #cmd_env['CILK_NWORKERS'] = '%d' % (WORKERS_PER_CPU,)
            
            jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
            
            n_jobs += 1
            
    
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs

    print 'WS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        


def CNN_execute(blocks):

    print 'CNN_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_FC_DNN_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    for block in blocks:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        cnn_block_n_depth = BLOCK_N_DEPTH
        if IS_3D_FIX:
            if block_depth > 0:
                cnn_block_n_depth += 2
        
        cnn_block_n_rows = BLOCK_N_ROWS
        cnn_block_n_cols = BLOCK_N_COLS
        if block_row_id > 0:
            cnn_block_n_rows += CNN_PATCH_LEG
        if block_row_id < BLOCKS_MAX_ROW_ID:
            cnn_block_n_rows += CNN_PATCH_LEG
        if block_col_id > 0:
            cnn_block_n_cols += CNN_PATCH_LEG
        if block_col_id < BLOCKS_MAX_COL_ID:
            cnn_block_n_cols += CNN_PATCH_LEG
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)        
        block_path_probs = os.path.join(BLOCKS_DIR_PROBS, '%s_probs' % (block_name,))
        
        if not is_block_valid(block_depth, block_row_id, block_col_id):
            continue
            
        segments = get_segments(block_path)
        
        for (seg_name, n_seg_files) in segments:
            seg_path = os.path.join(block_path, seg_name)
            seg_path_probs = os.path.join(block_path_probs, '%s_probs' % (seg_name,))
            
            is_overwrite = 0
            if not os.path.exists(seg_path_probs):
                os.makedirs(seg_path_probs)
            else:
                is_overwrite = 1
            
            print ' -- %s %s [%d,%d,%d] [overwrite = %d]' % (block_name, seg_name, n_seg_files, cnn_block_n_rows, cnn_block_n_cols, is_overwrite)
            print ' -- INPUT  : %s' % (seg_path,)
            print ' -- OUTPUT : %s' % (seg_path_probs,)
            
            fc_dnn_cmd = ('%s 2 %d %d %d %d %d 1 %d %s %s' % 
                (FC_DNN_PATH,
                 CNN_PATCH_LEG,
                 FC_DNN_3D_DEPTH,
                 n_seg_files,
                 cnn_block_n_rows,
                 cnn_block_n_cols,
                 FC_DNN_N_CHANNELS,
                 seg_path,
                 seg_path_probs))
            
            numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
            
            final_cmd = '%s %s' % (numactl_cmd, fc_dnn_cmd)
            
            log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_probs.log' % (seg_name,))
            cmd_env = os.environ.copy()
            cmd_env['CILK_NWORKERS'] = '%d' % (N_PER_FC_DNN_WORKERS,)
            
            jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
            
            n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'CNN_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        

def PROBS_COMBINE_execute(blocks):
    
    print 'PROBS_COMBINE_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_PROBS_COMBINE_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    for block in blocks:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        cnn_block_n_depth = BLOCK_N_DEPTH
        if IS_3D_FIX:
            if block_depth > 0:
                cnn_block_n_depth += 2
        
        cnn_block_n_rows = BLOCK_N_ROWS - (CNN_PATCH_LEG * 2)
        cnn_block_n_cols = BLOCK_N_COLS - (CNN_PATCH_LEG * 2)
        if block_row_id > 0:
            cnn_block_n_rows += CNN_PATCH_LEG
        if block_row_id < BLOCKS_MAX_ROW_ID:
            cnn_block_n_rows += CNN_PATCH_LEG
        if block_col_id > 0:
            cnn_block_n_cols += CNN_PATCH_LEG
        if block_col_id < BLOCKS_MAX_COL_ID:
            cnn_block_n_cols += CNN_PATCH_LEG
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)        
        block_path_probs = os.path.join(BLOCKS_DIR_PROBS, '%s_probs' % (block_name,))
        block_path_probs_combined = os.path.join(BLOCKS_DIR_PROBS_COMBINED, '%s_probs' % (block_name,))
        
        if not is_block_valid(block_depth, block_row_id, block_col_id):
            continue
        
        segments = get_segments(block_path)
        
        for (seg_name, n_seg_files) in segments:
            seg_path = os.path.join(block_path, seg_name)
            seg_path_probs = os.path.join(block_path_probs, '%s_probs' % (seg_name,))
            seg_path_probs_combined = os.path.join(block_path_probs_combined, '%s_probs' % (seg_name,))
            
            is_overwrite = 0
            if not os.path.exists(seg_path_probs_combined):
                os.makedirs(seg_path_probs_combined)
            else:
                is_overwrite = 1
            
            print ' -- %s %s [%d,%d,%d] [overwrite = %d]' % (block_name, seg_name, n_seg_files, cnn_block_n_rows, cnn_block_n_cols, is_overwrite)
            print ' -- INPUT  : %s' % (seg_path_probs,)
            print ' -- OUTPUT : %s' % (seg_path_probs_combined,)
             
            probs_combine_cmd = ('%s %s %s %d %f %f %d %d %d' % 
                (PROBS_COMBINE_PATH,
                 seg_path_probs, # input
                 seg_path_probs_combined, # output
                 PROBS_COMBINE_KERNEL_SIZE,
                 PROBS_COMBINE_REGIONAL_DEPTH_REDUCE,
                 PROBS_COMBINE_STD_SMOOTH,
                 cnn_block_n_rows,
                 cnn_block_n_cols,
                 n_seg_files))
             
            numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
            
            final_cmd = '%s %s' % (numactl_cmd, probs_combine_cmd)
            
            log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_probs_combine.log' % (seg_name,))
            cmd_env = os.environ.copy()
            
            jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
            
            n_jobs += 1
            
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'PROBS_COMBINE_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)
    

def WS_NP_PREPARE_execute(blocks):

    print 'WS_NP_PREPARE_execute: start'

    start_time_secs = time.time()

    max_jobs = N_WS_NP_PREPARE_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)        
        block_path_probs = os.path.join(BLOCKS_DIR_PROBS, '%s_probs' % (block_name,))
        
        if IS_WS_NP_PREPARE_USE_COMBINED_PROBS:
            block_path_probs_combined = os.path.join(BLOCKS_DIR_PROBS_COMBINED, '%s_probs' % (block_name,))
        
        block_path_probs_ws = os.path.join(BLOCKS_DIR_PROBS_WS, '%s_probs' % (block_name,))
        block_path_probs_np = os.path.join(BLOCKS_DIR_PROBS_NP, '%s_probs' % (block_name,))
        block_path_meta = os.path.join(BLOCKS_DIR_META, '%s_meta' % (block_name,))        
        block_path_meta_all = os.path.join(BLOCKS_DIR_META, 'all_blocks_%.4d_%.4d_meta' % (block_row_id, block_col_id))        
        
        if not is_block_valid(block_depth, block_row_id, block_col_id):
            continue
        
        all_block_outer_mask_filename = os.path.join(block_path_meta_all, OUTER_MASK_FILENAME % (block_row_id, block_col_id))
        all_block_border_mask_filename = os.path.join(block_path_meta_all, BORDER_MASK_FILENAME % (block_row_id, block_col_id))
        
        if not os.path.exists(all_block_outer_mask_filename):
            raise Exception('File missing: %s' % (all_block_outer_mask_filename,))
        
        if not os.path.exists(all_block_border_mask_filename):
            raise Exception('File missing: %s' % (all_block_border_mask_filename,))
        
        segments = get_segments(block_path)
        
        for (seg_name, n_seg_files) in segments:
            seg_path = os.path.join(block_path, seg_name)
            
            if IS_WS_NP_PREPARE_USE_COMBINED_PROBS:
                seg_path_probs = os.path.join(block_path_probs_combined, '%s_probs' % (seg_name,))
            else:
                seg_path_probs = os.path.join(block_path_probs, '%s_probs' % (seg_name,))
            
            seg_path_probs_ws = os.path.join(block_path_probs_ws, '%s_probs' % (seg_name,))
            seg_path_probs_np = os.path.join(block_path_probs_np, '%s_probs' % (seg_name,))
            seg_path_meta = os.path.join(block_path_meta, '%s_meta' % (seg_name,))
            
            is_overwrite_ws = 0
            if not os.path.exists(seg_path_probs_ws):
                os.makedirs(seg_path_probs_ws)
            else:
                is_overwrite_ws = 1
            
            is_overwrite_np = 0
            if not os.path.exists(seg_path_probs_np):
                os.makedirs(seg_path_probs_np)
            else:
                is_overwrite_np = 1
            
            print ' -- %s %s [%d] [overwrite_ws = %d, overwrite_np = %d]' % (
                block_name, 
                seg_name, 
                n_seg_files, 
                is_overwrite_ws, 
                is_overwrite_np)
            
            print ' -- INPUT_NP    : %s' % (seg_path_probs,)
            print ' -- OUTPUT_WS   : %s' % (seg_path_probs_ws,)
            print ' -- OUTPUT_NP   : %s' % (seg_path_probs_np,)
            
            block_outer_mask_filename = os.path.join(seg_path_meta, BLOCK_OUTER_MASK_FILENAME)
            block_border_mask_filename = os.path.join(seg_path_meta, BLOCK_BORDER_MASK_FILENAME)
            
            if IS_Z_DEPTH_CUT:
                block_outer_mask_filename = all_block_outer_mask_filename
                block_border_mask_filename = all_block_border_mask_filename
                
            ws_np_prepare_cmd = ('%s %d %s %s %s %s %s %s' % 
                (WS_NP_PREPARE_PATH,
                 CNN_PATCH_LEG,
                 block_outer_mask_filename,
                 block_border_mask_filename,
                 seg_path_meta,
                 seg_path_probs,
                 seg_path_probs_ws,
                 seg_path_probs_np))
            
            numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
            
            final_cmd = '%s %s' % (numactl_cmd, ws_np_prepare_cmd)
            
            log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_ws_np_prepare.log' % (seg_name,))
            cmd_env = os.environ.copy()
            
            jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
            
            n_jobs += 1
            
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'WS_NP_PREPARE_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        
    

def PREPARE_DATA_SUBBLOCKS_execute(blocks):
    
    print 'PREPARE_DATA_SUBBLOCKS_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_PREPARE_DATA_SUBBLOCKS_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_id = '%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_name = 'block_%s' % (block_id,)
        block_path_em_cc = os.path.join(BLOCKS_DIR_EM_CC, block_name)        
        block_path_meta = os.path.join(block_path_em_cc, 'blacked-with-patches-blacked', 'patches')
        
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
        
        is_overwrite = 0
        if not os.path.exists(block_path):
            os.makedirs(block_path)
        else:
            is_overwrite = 1
        
        if not is_block_exists(block_depth, block_row_id, block_col_id):
            continue
        
        print ' -- ID     : %s [overwrite = %d]' % (block_name, is_overwrite)
        print ' -- INPUT  : %s' % (block_path_em_cc,)
        print ' -- OUTPUT : %s' % (block_path,)
        
        prepare_data_subblocks_cmd = ('%s %d %s %s %s %s' % 
                (PREPARE_DATA_SUBBLOCKS_PATH,
                 PREPARE_DATA_SUBBLOCKS_IS_IGNORE,
                 block_id,
                 block_path_em_cc,
                 block_path_meta,
                 block_path))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, prepare_data_subblocks_cmd)
        
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_prepare_data_subblocks.log' % (block_name,))
        cmd_env = os.environ.copy()
        #cmd_env['CILK_NWORKERS'] = '%d' % (N_FC_DNN_WORKERS_PER_CPU,)
        
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'PREPARE_DATA_SUBBLOCKS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        
  
def PREPARE_DATA_BLACKEN_execute(blocks):
    
    print 'PREPARE_DATA_BLACKEN_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_PREPARE_DATA_BLACKEN_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_id = '%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_name = 'block_%s' % (block_id,)
        block_path_em_cc = os.path.join(BLOCKS_DIR_EM_CC, block_name)        
        
        if not os.path.exists(block_path_em_cc):
            raise Exception('ERROR: No directory: %s' % (block_path_em_cc))
        
        if not is_block_exists(block_depth, block_row_id, block_col_id):
            continue
        
        print ' -- ID     : %s' % (block_name,)
        print ' -- INPUT  : %s' % (block_path_em_cc,)
        print ' -- OUTPUT : %s' % (block_path_em_cc,)
        
        prepare_data_blacken_cmd = ('%s %s' % 
            (PREPARE_DATA_BLACKEN_PATH,
             block_path_em_cc))
             
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, prepare_data_blacken_cmd)
        
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_prepare_data_blacken.log' % (block_name,))
        cmd_env = os.environ.copy()
        #cmd_env['CILK_NWORKERS'] = '%d' % (N_FC_DNN_WORKERS_PER_CPU,)
        
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'PREPARE_DATA_BLACKEN_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        
    
def BORDER_MASKS_execute(blocks):
    
    print 'BORDER_MASKS_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_BORDER_MASKS_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
        block_path_meta = os.path.join(BLOCKS_DIR_META, '%s_meta' % (block_name,))
        
        if not is_block_exists(block_depth, block_row_id, block_col_id):
            continue
        
        segments = get_segments(block_path)
        
        for (seg_name, n_seg_files) in segments:
            seg_path = os.path.join(block_path, seg_name)
            seg_path_meta = os.path.join(block_path_meta, '%s_meta' % (seg_name,))
            
            if not os.path.exists(seg_path_meta):
                os.makedirs(seg_path_meta)
            
            print ' -- %s %s [%d]' % (block_name, seg_name, n_seg_files)
            print ' -- INPUT  : %s' % (seg_path,)
            print ' -- OUTPUT : %s' % (seg_path_meta,)
            
            border_masks_cmd = ('%s %d %d %s %s' % 
                (BORDER_MASKS_PATH,
                 BORDER_WIDTH,
                 CLOSE_WIDTH,
                 seg_path,
                 seg_path_meta))
            
            numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
            
            final_cmd = '%s %s' % (numactl_cmd, border_masks_cmd)
            
            log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_border_masks.log' % (seg_name,))
            cmd_env = os.environ.copy()
            
            jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
            
            n_jobs += 1
            
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'BORDER_MASKS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        


def COMBINE_BORDER_MASKS_execute(blocks):
    
    print 'COMBINE_BORDER_MASKS_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_COMBINE_BORDER_MASKS_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    done_list = []
    for block in blocks:
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
        block_path_meta = os.path.join(BLOCKS_DIR_META, '%s_meta' % (block_name,))
        
        if not is_block_exists(block_depth, block_row_id, block_col_id):
            continue
            
        if (block_row_id, block_col_id) in done_list:
            print 'SKIP %s: already done' % (block_name,)
            continue
            
        done_list.append((block_row_id, block_col_id))
        
        segments = get_segments(block_path)
        
        print ' -- %s ' % (block_name,)
        print ' -- INPUT  : %s [%d %d]' % (BLOCKS_DIR_META, block_row_id, block_col_id)
        print ' -- OUTPUT : %s' % (BLOCKS_DIR_META,)
        
        combine_border_masks_cmd = ('%s %s %d %d' % 
            (COMBINE_BORDER_MASKS_PATH,
             BLOCKS_DIR_META,
             block_row_id,
             block_col_id))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, combine_border_masks_cmd)
        
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_combine_border_masks.log' % (block_name,))
        cmd_env = os.environ.copy()
        
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1

    jobs_sync(max_jobs, procs, procs_data, True)

    elapsed_time_secs = time.time() - start_time_secs

    print 'COMBINE_BORDER_MASKS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)
    

def PREPARE_DATA_CC_execute(blocks):
    
    print 'PREPARE_DATA_CC_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_PREPARE_DATA_CC_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_id = '%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_name = 'block_%s' % (block_id,)
        block_path_em = os.path.join(BLOCKS_DIR_EM, block_name)        
        block_path_em_cc = os.path.join(BLOCKS_DIR_EM_CC, block_name)
        
        is_overwrite = 0
        if not os.path.exists(block_path_em_cc):
            os.makedirs(block_path_em_cc)
        else:
            is_overwrite = 1
            
        print ' -- ID     : %s [overwrite = %d]' % (block_name, is_overwrite)
        print ' -- INPUT  : %s' % (block_path_em,)
        print ' -- OUTPUT : %s' % (block_path_em_cc,)
        
        prepare_data_cc_cmd = ('%s %s %s' % 
                    (PREPARE_DATA_CC_PATH,
                     block_path_em,
                     block_path_em_cc))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, prepare_data_cc_cmd)
        
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_prepare_data_cc.log' % (block_name,))
        cmd_env = os.environ.copy()
        #cmd_env['CILK_NWORKERS'] = '%d' % (N_FC_DNN_WORKERS_PER_CPU,)
        
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'PREPARE_DATA_CC_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        
    
def DEBUG_GENERATE_exec(blocks):
    
    print 'DEBUG_GENERATE_exec: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_DEBUG_GENERATE_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in blocks:
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_n_depth = BLOCK_N_DEPTH
        if IS_3D_FIX:
            if block_depth == 0:
                block_n_depth -= 2
        
        block_n_rows = BLOCK_N_ROWS
        block_n_cols = BLOCK_N_COLS
        if block_row_id == 0:
            block_n_rows -= CNN_PATCH_LEG
        if block_row_id == BLOCKS_MAX_ROW_ID:
            block_n_rows -= CNN_PATCH_LEG
        if block_col_id == 0:
            block_n_cols -= CNN_PATCH_LEG
        if block_col_id == BLOCKS_MAX_COL_ID:
            block_n_cols -= CNN_PATCH_LEG
        
        block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
        block_path = os.path.join(BLOCKS_DIR_INPUT, block_name)
        
        if not is_block_valid(block_depth, block_row_id, block_col_id):
            continue
        
        print ' -- %s [%d,%d,%d]' % (block_name, 
                                     block_n_depth, 
                                     block_n_rows, 
                                     block_n_cols)
        
        debug_generate_cmd = ('%s %s %s %s %s %s %s %s %s %s %d %d %d %d %d %d %d' %
            (DEBUG_GENERATE_PATH,
             BLOCKS_DIR_INPUT, 
             BLOCKS_DIR_PROBS_COMBINED,
             BLOCKS_DIR_PROBS_WS,
             BLOCKS_DIR_PROBS_NP,
             BLOCKS_DIR_WS,
             BLOCKS_DIR_NP,
             BLOCKS_DIR_MERGE,
             BLOCKS_DIR_DEBUG, 
             DEBUG_OUT_PREFIX,
             block_depth, 
             block_row_id, 
             block_col_id, 
             block_n_depth,
             block_n_rows,
             block_n_cols,
             CNN_PATCH_LEG,))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, debug_generate_cmd)
        
        log_filepath = os.path.join(BLOCKS_DIR_LOGS, '%s_debug_generate.log' % (block_name,))
        cmd_env = os.environ.copy()
        
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'DEBUG_GENERATE_exec: finish [total_time = %d secs]' % (elapsed_time_secs,)
    

def execute(phase_id):
    
    if phase_id == PHASE_ALIGN_GENERATE_TILES:
        ALIGN_GENERATE_TILES_execute()
    
    elif phase_id == PHASE_ALIGN_GENERATE_TILES_TXT:
            ALIGN_GENERATE_TILES_TXT_execute()
    
    elif phase_id == PHASE_ALIGN_COMPUTE_KPS_AND_MATCH:
        ALIGN_COMPUTE_KPS_AND_MATCH_execute()
    
    elif phase_id == PHASE_ALIGN_COMPUTE_TRANSFORMS:
        ALIGN_COMPUTE_TRANSFORMS_execute()
    
    elif phase_id == PHASE_ALIGN_COMPUTE_WARPS:
        ALIGN_COMPUTE_WARPS_execute()
    
    elif phase_id == PHASE_PREPARE_DATA_CC:
        PREPARE_DATA_CC_execute(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_PREPARE_DATA_BLACKEN:
        PREPARE_DATA_BLACKEN_execute(BLOCKS_TO_PROCESS)
        
    elif phase_id == PHASE_PREPARE_DATA_SUBBLOCKS:
        PREPARE_DATA_SUBBLOCKS_execute(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_BORDER_MASKS:
        BORDER_MASKS_execute(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_COMBINE_BORDER_MASKS:
        COMBINE_BORDER_MASKS_execute(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_CNN:
        CNN_execute(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_PROBS_COMBINE:
        PROBS_COMBINE_execute(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_WS_NP_PREPARE:
        WS_NP_PREPARE_execute(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_WS:
        WS_execute(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_NP_PREPARE:
        NP_execute_prepare(BLOCKS_TO_PROCESS)
        
    elif phase_id == PHASE_NP_EXEC:
        NP_execute(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_BLOCK_VIDEO:
        block_video(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_MERGE_PREPROCESS:
        MERGE_preprocess(BLOCKS_TO_PROCESS)
        
    elif phase_id == PHASE_MERGE_EXEC:
        MERGE_exec(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_MERGE_COMBINE:
        MERGE_combine(BLOCKS_TO_PROCESS)
          
    elif phase_id == PHASE_MERGE_RELABEL:
        MERGE_relabel(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_SKELETONS:
        skeleton_exec(BLOCKS_TO_PROCESS)
    
    elif phase_id == PHASE_DEBUG_GENERATE:
        DEBUG_GENERATE_exec(BLOCKS_TO_PROCESS)
        
    else:
        raise Exception('Unexpected phase [%s]' % (phase_id,))
    

def phase_num_to_str(phase_num):
    
    phase_num = int(phase_num) - 1
    
    if ((phase_num < 0) or (phase_num >= len(PHASE_LIST))):
        raise Exception('Unexpected phase number [%d]' % (phase_num,))    
    
    return PHASE_LIST[phase_num]
    

if '__main__' == __name__:
    try:
        prog_name, phase_id = sys.argv[:2]
        
        if phase_id.isdigit():
            phase_id = phase_num_to_str(phase_id)
        
    except ValueError, e:
        sys.exit('USAGE: %s [phase]' % (sys.argv[0],))

    
    execute(phase_id)

