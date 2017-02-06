
import sys
import os
import argparse
import shutil
import time
import subprocess
from imread import imread,imsave

import cv2
import numpy as np

from pipeline_common import *

# General params

IS_WS_NP_PREPARE = 1
IS_WS_NP_PREPARE_USE_COMBINED_PROBS = 1

###############################################################################
# ALIGN general params
###############################################################################
N_ALIGN_GENERATE_BASE_SECTION = 0
N_ALIGN_GENERATE_SECTIONS = 1850
N_ALIGN_GENERATE_BATCH = 8

N_ALIGN_EXEC_BASE_SECTION = 0
N_ALIGN_EXEC_SECTIONS = 1850
N_ALIGN_EXEC_BATCH = 4

ALIGN_BIN_PATH = '/home/armafire/Pipeline/align/align_proto/run_align'

###############################################################################
# ALIGN_GENERATE_TILES params
###############################################################################
N_ALIGN_GENERATE_TILES_WORKERS_PER_CPU = 9

ALIGN_EM_SOURCE = '/mnt/disk1/armafire/datasets/K11_orig_from_DanielBerger' 

ALIGN_TILES_BIN_PATH = 'python /home/armafire/Pipeline/align/align_proto/scripts/K11_S1_to_tiles.py'

ALIGN_TILES_DIR = 'tiles'

###############################################################################
# ALIGN_GENERATE_TILES_TXT params
###############################################################################

ALIGN_TILES_TXT_BIN_PATH = 'python /home/armafire/Pipeline/align/align_proto/scripts/K11_tiles_to_txt_input.py'

ALIGN_TILES_TXT_DIR = os.path.join(META_DIR, ALIGN_DIR, 'tiles_txt')

ALIGN_TILES_TXT_FILEPATH = os.path.join(
    ALIGN_TILES_TXT_DIR, 
    'K11_S1_3nm_tiles_txt_0-4_0-4.txt')

###############################################################################
# ALIGN_COMPUTE_KPS_AND_MATCH params
###############################################################################

N_ALIGN_COMPUTE_KPS_AND_MATCH_WORKERS_PER_CPU = 4
N_ALIGN_COMPUTE_KPS_AND_MATCH_CPU_WORKERS = 9
N_ALIGN_COMPUTE_KPS_AND_MATCH_OMP_NUM_THREADS = 9

###############################################################################
# ALIGN_COMPUTE_WARPS params
###############################################################################

N_ALIGN_COMPUTE_WARPS_WORKERS_PER_CPU = 9
N_ALIGN_COMPUTE_WARPS_CPU_WORKERS = 2

###############################################################################
# PROBS_COMBINE params
###############################################################################

N_PROBS_COMBINE_WORKERS_PER_CPU = 8 * 2

PROBS_COMBINE_PATH = '/home/armafire/Pipeline/matlab_scripts/run_extractSeeds.sh'

PROBS_COMBINE_KERNEL_SIZE = 13
PROBS_COMBINE_REGIONAL_DEPTH_REDUCE = 0.04
PROBS_COMBINE_STD_SMOOTH = 2.0

###############################################################################
# BLOCK_VIDEO params
###############################################################################

BLOCK_VIDEO_PATH = '/home/armafire/Pipeline/matlab_scripts/run_mergeH5_1core.sh'

N_BLOCK_VIDEO_WORKERS_PER_CPU = 8

IS_BLOCK_VIDEO = 1

###############################################################################
# SCATTER_POINTS params
###############################################################################

N_SCATTER_POINTS_WORKERS_PER_CPU = 32

SCATTER_POINTS_BIN_PATH = '/mnt/disk1/hayk/block-operations/run_scatter_points.sh'

###############################################################################
# SKELETON params
###############################################################################

N_SKELETONS_WORKERS_PER_CPU = 8

N_PER_SKELETON_EXEC_CILK_WORKERS = 1#4

SKELETON_PATH = '/home/armafire/Pipeline/pipeline_engine/skeletonization/graph_extraction/run_main.sh'

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
    cur_section = N_ALIGN_GENERATE_BASE_SECTION
    finish_section = N_ALIGN_GENERATE_BASE_SECTION + N_ALIGN_GENERATE_SECTIONS
    
    tiles_path_list = []
    for data_dir in DATA_DIR_LIST:
        tiles_path = os.path.join(data_dir, ALIGN_DIR, ALIGN_TILES_DIR)
        tiles_path_list.append(tiles_path)
        
        if not os.path.exists(tiles_path):
            os.makedirs(tiles_path)
    
    while cur_section < finish_section:
        
        base_section = cur_section
        n_sections = 1
        cur_section += 1
        
        if cur_section > finish_section:
            cur_section = finish_section
        
        data_idx = base_section % len(tiles_path_list)
        
        print ' -- ALIGN_GENERATE_TILES: [%d -> %d] ' % (base_section, base_section + n_sections)
        
        align_generate_tiles_cmd = ('%s %s %d %d %s' % 
            (ALIGN_TILES_BIN_PATH,
             ALIGN_EM_SOURCE,
             base_section,
             n_sections,
             tiles_path_list[data_idx]))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, align_generate_tiles_cmd)
        
        log_filepath = os.path.join(
            META_DIR, LOG_DIR, ALIGN_DIR, 'align_generate_tiles_%.4d_%.4d.log' % (base_section, n_sections))
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
    
    if not os.path.exists(ALIGN_TILES_TXT_DIR):
        os.makedirs(ALIGN_TILES_TXT_DIR)
    
    align_generate_tiles_txt_cmd = ('%s %s %d %d %s' % 
        (ALIGN_TILES_TXT_BIN_PATH,
         'NONE',
         N_ALIGN_GENERATE_BASE_SECTION,
         N_ALIGN_GENERATE_SECTIONS,
         ALIGN_TILES_TXT_FILEPATH))
    
    numactl_cmd = get_numa_cmd(0)
    
    final_cmd = '%s %s' % (numactl_cmd, align_generate_tiles_txt_cmd)
    
    log_filepath = os.path.join(
        META_DIR, LOG_DIR, ALIGN_DIR, 'align_generate_tiles_txt_%.4d_%.4d.log' % 
        (N_ALIGN_GENERATE_BASE_SECTION, N_ALIGN_GENERATE_BASE_SECTION + N_ALIGN_GENERATE_SECTIONS))
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
    cur_section = N_ALIGN_EXEC_BASE_SECTION
    
    work_dir_path = os.path.join(META_DIR, ALIGN_DIR, ALIGN_WORK_DIR)
    
    if not os.path.exists(work_dir_path):
        os.makedirs(work_dir_path)
        
    while cur_section < (N_ALIGN_EXEC_BASE_SECTION + N_ALIGN_EXEC_SECTIONS):
        
        base_section = cur_section
        n_sections = N_ALIGN_EXEC_BATCH
        cur_section += (N_ALIGN_EXEC_BATCH - 1)
        
        print ' -- ALIGN_COMPUTE_KPS_AND_MATCH: [%d -> %d] ' % (base_section, base_section + n_sections)
        
        if ((base_section + n_sections) > (N_ALIGN_EXEC_BASE_SECTION + N_ALIGN_EXEC_SECTIONS)):
            n_sections = (N_ALIGN_EXEC_BASE_SECTION + N_ALIGN_EXEC_SECTIONS) - base_section
        
        if (n_sections == 1):
            continue
        
        align_compute_kps_and_match_cmd = ('%s 1 %d %d %s %s %s' % 
            (ALIGN_BIN_PATH,
             base_section,
             n_sections,
             ALIGN_TILES_TXT_FILEPATH,
             work_dir_path,
             'NONE'))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, align_compute_kps_and_match_cmd)
        
        log_filepath = os.path.join(
            META_DIR, LOG_DIR, ALIGN_DIR, 'align_compute_kps_and_match_%.4d_%.4d.log' % 
            (base_section, n_sections))
        cmd_env = os.environ.copy()
        cmd_env['CILK_NWORKERS'] = '%d' % (N_ALIGN_COMPUTE_KPS_AND_MATCH_CPU_WORKERS,)
        cmd_env['OMP_NUM_THREADS'] = '%d' % (N_ALIGN_COMPUTE_KPS_AND_MATCH_OMP_NUM_THREADS,)
        
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
    cur_section = N_ALIGN_EXEC_BASE_SECTION
    
    work_dir_path = os.path.join(META_DIR, ALIGN_DIR, ALIGN_WORK_DIR)
    
    print ' -- ALIGN_COMPUTE_TRANSFORMS: [%d -> %d] ' % (
        N_ALIGN_EXEC_BASE_SECTION, 
        N_ALIGN_EXEC_BASE_SECTION + N_ALIGN_EXEC_SECTIONS)
    
    align_compute_transforms_cmd = ('%s 2 %d %d %s %s %s' % 
        (ALIGN_BIN_PATH,
         N_ALIGN_GENERATE_BASE_SECTION,
         N_ALIGN_GENERATE_SECTIONS,
         ALIGN_TILES_TXT_FILEPATH,
         work_dir_path,
         'NONE'))
    
    numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
    
    final_cmd = '%s %s' % (numactl_cmd, align_compute_transforms_cmd)
    
    log_filepath = os.path.join(
        META_DIR, LOG_DIR, ALIGN_DIR, 'align_compute_transforms_%.4d_%.4d.log' % 
        (N_ALIGN_EXEC_BASE_SECTION, N_ALIGN_EXEC_BASE_SECTION + N_ALIGN_EXEC_SECTIONS))
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
    cur_section = N_ALIGN_EXEC_BASE_SECTION
    
    work_dir_path = os.path.join(META_DIR, ALIGN_DIR, ALIGN_WORK_DIR)
        
    while cur_section < (N_ALIGN_EXEC_BASE_SECTION + N_ALIGN_EXEC_SECTIONS):
        
        base_section = cur_section
        n_sections = 1
        cur_section += 1
        
        if ((base_section + n_sections) > (N_ALIGN_EXEC_BASE_SECTION + N_ALIGN_EXEC_SECTIONS)):
            n_sections = (N_ALIGN_EXEC_BASE_SECTION + N_ALIGN_EXEC_SECTIONS) - base_section
        
        result_dir_path = get_align_result_dir(base_section)
        
        if not os.path.exists(result_dir_path):
            os.makedirs(result_dir_path)
        
        print ' -- ALIGN_COMPUTE_WARPS: [%d -> %d] ' % (base_section, base_section + n_sections)
        
        align_compute_warps_cmd = ('%s 3 %d %d %s %s %s' % 
            (ALIGN_BIN_PATH,
             base_section,
             n_sections,
             ALIGN_TILES_TXT_FILEPATH,
             work_dir_path,
             result_dir_path))
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
        
        final_cmd = '%s %s' % (numactl_cmd, align_compute_warps_cmd)
        
        log_filepath = os.path.join(
            META_DIR, LOG_DIR, ALIGN_DIR, 'align_compute_warps_%.4d_%.4d.log' % (base_section, n_sections))
        cmd_env = os.environ.copy()
        cmd_env['CILK_NWORKERS'] = '%d' % (N_ALIGN_COMPUTE_WARPS_CPU_WORKERS,)
        
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
        
        n_jobs += 1
        
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ALIGN_COMPUTE_WARPS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        

#####
def NP_PREPARE_execute():
    
    print 'NP_PREPARE_execute: start'
    
    start_time_secs = time.time()
        
    max_jobs = N_NP_PREPARE_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in BLOCKS_TO_PROCESS:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not meta_is_block_valid(
            block_depth, 
            block_row_id, 
            block_col_id):
            print ' -- %s is not valid [SKIP]' % (block_name,)
            continue
                
        block_probs_np_path = get_block_probs_np_path(    
            block_depth, 
            block_row_id, 
            block_col_id)
        
        block_ws_path = get_block_ws_path(    
            block_depth, 
            block_row_id, 
            block_col_id)
        
        block_np_path = get_block_np_path(    
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not os.path.exists(block_np_path):
            os.makedirs(block_np_path)    
        else:
            clean_dir(block_np_path)
        
        print ' -- %s' % (block_name,)
            
        print ' -- INPUT  : %s' % (block_ws_path,)
        print ' -- OUTPUT : %s' % (block_np_path,)
        
        np_prepare_cmd = ('python np_generate_inputs.py %d %s %s %s %s' %
            (NP_INPUTS_IS_OVERWRITE,
             block_probs_np_path,
             block_ws_path,
             block_np_path,
             block_name))
            
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
            
        final_cmd = '%s %s' % (numactl_cmd, np_prepare_cmd)
        
        log_filepath = os.path.join(
            META_DIR, LOG_DIR, NP_DIR, '%s_np_prepare.log' % (block_name,))        
        cmd_env = os.environ.copy()
            
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
            
        n_jobs += 1
    
    jobs_sync(max_jobs, procs, procs_data, True)
        
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'NP_PREPARE_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)

def NP_verify(log_path):
    res = is_run_log_success(log_path)
    return res
    
def NP_execute(ctx):
    
    print 'NP_execute: start'
    
    phase_name = 'np'
    
    start_time_secs = time.time()
    
    np_classifier_filepath = verify_and_get_np_classifier(NP_CLASSIFIER_DIR)
    
    ctx.jobs.init(
        max_jobs = N_NP_EXEC_WORKERS_PER_CPU * N_CPUS)
    
    ctx.jobs.set_proc_verify_func(NP_verify)
        
    for block in ctx.blocks_to_process:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not meta_is_block_valid(
            block_depth, 
            block_row_id, 
            block_col_id):
            print ' -- %s is not valid [SKIP]' % (block_name,)
            continue
        
        block_probs_np_path = get_block_probs_np_path(    
            block_depth, 
            block_row_id, 
            block_col_id)
        
        block_ws_path = get_block_ws_path(    
            block_depth, 
            block_row_id, 
            block_col_id)
                        
        block_np_path = get_block_np_path(    
            block_depth, 
            block_row_id, 
            block_col_id)
        
        verify_block_out_dir(block_np_path)
            
        block_np_seg_prefix = os.path.join(block_np_path, '%s_np_seg_' % (PREFIX,))
        
        print ' -- %s' % (block_name,)
        print ' -- INPUT  : %s' % (block_np_path,)
        print ' -- OUTPUT : %s' % (block_np_path,)
        
        np_exec_cmd = ('%s %d %d %d %s' % 
            (NP_EXEC_PATH,
             block_depth,
             block_row_id,
             block_col_id,
             np_classifier_filepath))
        
        proc_log_filepath = os.path.join(
            META_DIR, LOG_DIR, LOG_PROCS_DIR, phase_name, '%s_np.log' % (block_name,))        
            
        ctx.jobs.execute(
            cmd = np_exec_cmd,
            log_path = proc_log_filepath,
            cmd_env = None)
    
    ctx.jobs.sync_all()
    
    run_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_RUNS_DIR, phase_name, 'np_summary.log')        
    
    ctx.jobs.report_status(run_log_filepath)
    
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

def MERGE_PREPROCESS_verify(log_path):
    res = is_run_log_success(log_path)
    return res
    
def MERGE_PREPROCESS_exec(ctx):
    
    print 'MERGE_PREPROCESS_exec: start'
    
    phase_name = 'merge_preprocess'
    
    start_time_secs = time.time()
    
    ctx.jobs.init(
        max_jobs = N_MERGE_PREPROCESS_WORKERS_PER_CPU * N_CPUS)
    
    ctx.jobs.set_proc_verify_func(MERGE_PREPROCESS_verify)
    
    merge_data_paths = get_merge_data_paths()
    for merge_data_dir in merge_data_paths:
        verify_out_dir(
            ctx, 
            merge_data_dir)
    
    for block in ctx.blocks_to_process:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not meta_is_block_valid(
            block_depth, 
            block_row_id, 
            block_col_id):
            print ' -- %s is not valid [SKIP]' % (block_name,)
            continue
        
        block_merge_path = get_block_merge_path(    
            block_depth, 
            block_row_id, 
            block_col_id)
        
        assert_out_dir(block_merge_path)
            
        print ' -- INPUT  : %s' % (block_name,)
        print ' -- OUTPUT : %s' % (block_merge_path,)
                
        merge_preprocess = ('%s %d %d %d' % 
            (MERGE_PREPROCESS_PATH,
             block_depth,
             block_row_id,
             block_col_id))
        
        proc_log_filepath = os.path.join(
            META_DIR, LOG_DIR, LOG_PROCS_DIR, phase_name, '%s_merge_preprocess.log' % (block_name,))        
        
        cmd_env_obj = os.environ.copy()
        cmd_env_obj['CILK_NWORKERS'] = '%d' % (N_MERGE_PREPROCESS_EXEC_CILK_WORKERS_PER_RUN,)
        
        ctx.jobs.execute(
            cmd = merge_preprocess,
            log_path = proc_log_filepath,
            cmd_env = cmd_env_obj)
    
    ctx.jobs.sync_all()
    
    run_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_RUNS_DIR, phase_name, 'merge_preprocess_summary.log')        
    
    ctx.jobs.report_status(run_log_filepath)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'MERGE_PREPROCESS_exec: finish [total_time = %d secs]' % (elapsed_time_secs,)
    
def MERGE_BLOCK_verify(log_path):
    res = is_run_log_success(log_path)
    return res
    
def MERGE_BLOCK_exec(ctx):
    
    print 'MERGE_BLOCK_exec: start'
    
    phase_name = 'merge_block'
    
    start_time_secs = time.time()
    
    ctx.jobs.init(
        max_jobs = N_MERGE_BLOCK_EXEC_WORKERS_PER_CPU * N_CPUS)
    
    ctx.jobs.set_proc_verify_func(MERGE_BLOCK_verify)
    
    merge_meta_path = get_merge_meta_path()
        
    verify_out_dir(ctx, merge_meta_path)
            
    for block in ctx.blocks_to_process:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not meta_is_block_valid(
            block_depth, 
            block_row_id, 
            block_col_id):
            print ' -- %s is not valid [SKIP]' % (block_name,)
            continue
        
        print ' -- %s' % (block_name,)
        
        merge_block_exec = ('%s %d %d %d' % 
            (MERGE_BLOCK_EXEC_PATH,
             block_depth,
             block_row_id,
             block_col_id)) 
        
        proc_log_filepath = os.path.join(
            META_DIR, LOG_DIR, LOG_PROCS_DIR, phase_name, '%s_merge_block.log' % (block_name,))        
        
        cmd_env_obj = os.environ.copy()
        cmd_env_obj['CILK_NWORKERS'] = '%d' % (N_MERGE_BLOCK_EXEC_CILK_WORKERS_PER_RUN,)
        
        ctx.jobs.execute(
            cmd = merge_block_exec,
            log_path = proc_log_filepath,
            cmd_env = cmd_env_obj)    
        
    ctx.jobs.sync_all()
    
    run_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_RUNS_DIR, phase_name, 'merge_block_summary.log')        
    
    ctx.jobs.report_status(run_log_filepath)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'MERGE_BLOCK_exec: finish [total_time = %d secs]' % (elapsed_time_secs,)

def MERGE_COMBINE_verify(log_path):
    res = is_run_log_success(log_path)
    return res

def MERGE_COMBINE_exec(ctx):
    
    print 'MERGE_COMBINE_exec: start'
    
    phase_name = 'merge_combine'
    
    start_time_secs = time.time()
    
    ctx.jobs.init(
        max_jobs = 1)
    
    ctx.jobs.set_proc_verify_func(MERGE_COMBINE_verify)

    merge_meta_path = get_merge_meta_path()
        
    if not os.path.exists(merge_meta_path):
        raise Exception('ERROR: merge meta not found [%s]' % (merge_meta_path))    
    
    merge_combine = ('%s' % 
        (MERGE_COMBINE_EXEC_PATH,))
    
    proc_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_PROCS_DIR, phase_name, 
        '[%.4d_%.4d_%.4d]-[%.4d_%.4d_%.4d]_merge_combine.log' % (
        ctx.blocks_to_process[0][0],
        ctx.blocks_to_process[0][1],
        ctx.blocks_to_process[0][2],
        ctx.blocks_to_process[-1][0]+1,
        ctx.blocks_to_process[-1][1]+1,
        ctx.blocks_to_process[-1][2]+1))     
    
    ctx.jobs.execute(
        cmd = merge_combine,
        log_path = proc_log_filepath,
        cmd_env = None)
    
    ctx.jobs.sync_all()
    
    run_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_RUNS_DIR, phase_name, 'merge_combine_summary.log')        
    
    ctx.jobs.report_status(run_log_filepath)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'MERGE_COMBINE_exec: finish [total_time = %d secs]' % (elapsed_time_secs,)

def MERGE_RELABEL_verify(log_path):
    res = is_run_log_success(log_path)
    return res

def MERGE_RELABEL_exec(ctx):
    
    print 'MERGE_RELABEL_exec: start'
    
    phase_name = 'merge_relabel'
    
    start_time_secs = time.time()
    
    ctx.jobs.init(
        max_jobs = N_MERGE_RELABEL_WORKERS_PER_CPU * N_CPUS)
    
    ctx.jobs.set_proc_verify_func(MERGE_RELABEL_verify)
    
    merge_meta_path = get_merge_meta_path()
        
    if not os.path.exists(merge_meta_path):
        raise Exception('ERROR: merge meta not found [%s]' % (merge_meta_path))
        
    for block in ctx.blocks_to_process:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not meta_is_block_valid(
            block_depth, 
            block_row_id, 
            block_col_id):
            print ' -- %s is not valid [SKIP]' % (block_name,)
            continue
        
        block_merge_path = get_block_merge_path(
            block_depth,
            block_row_id,
            block_col_id)
        
        if not os.path.exists(block_merge_path):
            raise Exception('ERROR: not found %s' % (block_merge_path,))
        
        print ' -- %s' % (block_name,)
        
        merge_relabel = ('%s %s %s %d %d %d' % 
            (MERGE_RELABEL_EXEC_PATH,
             merge_meta_path,
             block_merge_path,
             block_depth,
             block_row_id,
             block_col_id))
        
        proc_log_filepath = os.path.join(
            META_DIR, LOG_DIR, LOG_PROCS_DIR, phase_name, '%s_merge_relabel.log' % (block_name,))        
        
        ctx.jobs.execute(
            cmd = merge_relabel,
            log_path = proc_log_filepath,
            cmd_env = None)
    
    ctx.jobs.sync_all()
    
    run_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_RUNS_DIR, phase_name, 'merge_relabel_summary.log')        
    
    ctx.jobs.report_status(run_log_filepath)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'MERGE_relabel: finish [total_time = %d secs]' % (elapsed_time_secs,)

def SCATTER_POINTS_execute():
    
    print 'SCATTER_POINTS_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_SCATTER_POINTS_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
        
    verify_scatter_points_dirs();
    
    scatter_points_path = get_scatter_points_path()
    
    for block in BLOCKS_TO_PROCESS:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not meta_is_block_valid(
            block_depth, 
            block_row_id, 
            block_col_id):
            print ' -- %s is not valid [SKIP]' % (block_name,)
            continue
        
        block_out_merge_path = get_block_out_merge_path(
            block_depth,
            block_row_id,
            block_col_id)
        
        print ' -- %s' % (block_name,)
        print ' -- INPUT: %s' % (block_out_merge_path,)
        
        if not os.path.exists(block_out_merge_path):
            raise Exception('ERROR: not found %s' % (block_out_merge_path,))
        
        scatter_points_cmd = '%s %d %d %d %s %s' % (
            SCATTER_POINTS_BIN_PATH,
            block_depth,
            block_row_id,
            block_col_id,
            block_out_merge_path,
            scatter_points_path)
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
            
        final_cmd = '%s %s' % (numactl_cmd, scatter_points_cmd)
        
        log_filepath = os.path.join(
            META_DIR, LOG_DIR, MERGE_DIR, '%s_scatter_points.log' % (block_name,))        
        cmd_env = os.environ.copy()
            
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
            
        n_jobs += 1
            
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'SCATTER_POINTS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)

#
def SKELETONS_execute():
    
    print 'SKELETONS_execute: start'
    
    start_time_secs = time.time()
    
    max_jobs = N_SKELETONS_WORKERS_PER_CPU * N_CPUS
    n_jobs = 0
    procs = []
    procs_data = []
    
    for block in BLOCKS_TO_PROCESS:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not meta_is_block_valid(
            block_depth, 
            block_row_id, 
            block_col_id):
            print ' -- %s is not valid [SKIP]' % (block_name,)
            continue
        
        block_out_merge_path = get_block_out_merge_path(
            block_depth,
            block_row_id,
            block_col_id)
            
        block_skeletons_path = get_block_skeletons_path(
            block_depth,
            block_row_id,
            block_col_id)
        
        print ' -- %s' % (block_name,)
        print ' -- INPUT: %s' % (block_out_merge_path,)
        
        if not os.path.exists(block_out_merge_path):
            raise Exception('ERROR: not found %s' % (block_out_merge_path,))
        
        skeletons_cmd = '%s %s %s' % (
            SKELETON_PATH,
            block_out_merge_path,
            block_skeletons_path)
        
        numactl_cmd = get_numa_cmd(n_jobs % N_CPUS)
            
        final_cmd = '%s %s' % (numactl_cmd, skeletons_cmd)
        
        log_filepath = os.path.join(
            META_DIR, LOG_DIR, MERGE_DIR, '%s_skeletons.log' % (block_name,))        
        cmd_env = os.environ.copy()
        cmd_env['CILK_NWORKERS'] = '%d' % (N_PER_SKELETON_EXEC_CILK_WORKERS,)    
        
        jobs_exec_process(max_jobs, procs, procs_data, n_jobs, final_cmd, log_filepath, cmd_env)
            
        n_jobs += 1
            
    jobs_sync(max_jobs, procs, procs_data, True)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'SKELETONS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)

def WS_verify(log_path):
    res = is_run_log_success(log_path)
    return res

def WS_execute(ctx):
    
    print 'WS_execute: start'

    phase_name = 'ws'
    
    start_time_secs = time.time()
    
    ctx.jobs.init(
        max_jobs = N_WS_WORKERS_PER_CPU * N_CPUS)
    
    ctx.jobs.set_proc_verify_func(WS_NP_PREPARE_verify)
      
    for block in ctx.blocks_to_process:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not meta_is_block_valid(
            block_depth, 
            block_row_id, 
            block_col_id):
            print ' -- %s is not valid [SKIP]' % (block_name,)
            continue
        
        block_probs_ws_path = get_block_probs_ws_path(    
            block_depth, 
            block_row_id, 
            block_col_id)
        
        block_ws_path = get_block_ws_path(    
            block_depth, 
            block_row_id, 
            block_col_id)
        
        verify_block_out_dir(block_ws_path)
        
        print ' -- %s' % (block_name,)    
        print ' -- INPUT  : %s' % (block_probs_ws_path,)
        print ' -- OUTPUT : %s' % (block_ws_path,)
        
        ws_cmd_opt = ''
        if IS_WS_SEEDS_2D:
            ws_cmd_opt += ' --labels-2D'
        
        if IS_WS_2D:
            ws_cmd_opt += ' --ws-2D'
        
        if IS_WS_USE_BG:
            ws_cmd_opt += ' --useBG'
            
        ws_cmd = ('%s %s --input-path=%s --output-path=%s' % 
            (WS_PATH,
             ws_cmd_opt,
             block_probs_ws_path,
             block_ws_path))
        
        proc_log_filepath = os.path.join(
            META_DIR, LOG_DIR, LOG_PROCS_DIR, phase_name, '%s_ws.log' % (block_name,))        
            
        ctx.jobs.execute(
            cmd = ws_cmd,
            log_path = proc_log_filepath,
            cmd_env = None)
    
    ctx.jobs.sync_all()
    
    run_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_RUNS_DIR, phase_name, 'ws_summary.log')        
    
    ctx.jobs.report_status(run_log_filepath)
    
    elapsed_time_secs = time.time() - start_time_secs

    print 'WS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        

def CNN_verify(log_path):
    res = is_run_log_success(log_path)
    return res
    
def CNN_execute(ctx):

    print 'CNN_execute: start'
    
    phase_name = 'cnn'
    
    start_time_secs = time.time()
    
    ctx.jobs.init(
        max_jobs = N_FC_DNN_WORKERS_PER_CPU * N_CPUS)
    
    ctx.jobs.set_proc_verify_func(CNN_verify)
    
    for block in ctx.blocks_to_process:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not meta_is_block_valid(
            block_depth, 
            block_row_id, 
            block_col_id):
            print ' -- %s is not valid [SKIP]' % (block_name,)
            continue
            
        block_depth_size = get_block_em_depth_size(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        block_row_size = get_block_em_row_size(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        block_col_size = get_block_em_col_size(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        block_em_path = get_block_em_path(    
            block_depth, 
            block_row_id, 
            block_col_id)
            
        block_probs_path = get_block_probs_path(    
            block_depth, 
            block_row_id, 
            block_col_id)
        
        verify_block_out_dir(block_probs_path)
        
        print ' -- %s [%d,%d,%d]' % (
            block_name, 
            block_depth_size, 
            block_row_size,
            block_col_size)
            
        print ' -- INPUT  : %s' % (block_em_path,)
        print ' -- OUTPUT : %s' % (block_probs_path,)
            
        fc_dnn_cmd = ('%s 2 %d %d %d %d %d 1 %d %s %s' % 
            (FC_DNN_PATH,
             CNN_PATCH_LEG,
             FC_DNN_3D_DEPTH,
             block_depth_size,
             block_row_size,
             block_col_size,
             FC_DNN_N_CHANNELS,
             block_em_path,
             block_probs_path))
        
        proc_log_filepath = os.path.join(
            META_DIR, LOG_DIR, LOG_PROCS_DIR, phase_name, '%s_cnn.log' % (block_name,))        
        
        cmd_env_obj = os.environ.copy()
        cmd_env_obj['CILK_NWORKERS'] = '%d' % (N_PER_FC_DNN_WORKERS,)
            
        ctx.jobs.execute(
            cmd = fc_dnn_cmd,
            log_path = proc_log_filepath,
            cmd_env = cmd_env_obj)
    
    ctx.jobs.sync_all()
    
    run_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_RUNS_DIR, phase_name, 'cnn_summary.log')        
    
    ctx.jobs.report_status(run_log_filepath)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'CNN_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        

def CNN_POSTPROCESS_verify(log_path):
    res = is_run_log_success(log_path)
    return res

def CNN_POSTPROCESS_execute(ctx):

    print 'CNN_POSTPROCESS_execute: start'
    
    phase_name = 'cnn_postprocess'
    
    start_time_secs = time.time()
    
    ctx.jobs.init(
        max_jobs = N_CNN_POSTPROCESS_WORKERS_PER_CPU * N_CPUS)
    
    ctx.jobs.set_proc_verify_func(CNN_POSTPROCESS_verify)
    
    for block in ctx.blocks_to_process:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not meta_is_block_valid(
            block_depth, 
            block_row_id, 
            block_col_id):
            print ' -- %s is not valid [SKIP]' % (block_name,)
            continue
        
        print ' -- %s' % (block_name,)
                    
        cnn_postprocess_cmd = ('%s %d %d %d' % 
            (CNN_POSTPROCESS_EXEC_PATH,
             block_depth,
             block_row_id,
             block_col_id))
        
        proc_log_filepath = os.path.join(
            META_DIR, LOG_DIR, LOG_PROCS_DIR, phase_name, '%s_cnn_postprocess.log' % (block_name,))        
                    
        ctx.jobs.execute(
            cmd = cnn_postprocess_cmd,
            log_path = proc_log_filepath,
            cmd_env = None)
    
    ctx.jobs.sync_all()
    
    run_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_RUNS_DIR, phase_name, 'cnn_postprocess_summary.log')        
    
    ctx.jobs.report_status(run_log_filepath)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'CNN_POSTPROCESS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        

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
        if CNN_IS_3D_FIX:
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
    
def WS_NP_PREPARE_verify(log_path):
    res = is_run_log_success(log_path)
    return res
    
def WS_NP_PREPARE_execute(ctx):

    print 'WS_NP_PREPARE_execute: start'
    
    phase_name = 'ws_np_prepare'
    
    start_time_secs = time.time()
    
    ctx.jobs.init(
        max_jobs = N_WS_NP_PREPARE_WORKERS_PER_CPU * N_CPUS)
    
    ctx.jobs.set_proc_verify_func(WS_NP_PREPARE_verify)
        
    for block in ctx.blocks_to_process:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not meta_is_block_valid(
            block_depth, 
            block_row_id, 
            block_col_id):
            print ' -- %s is not valid [SKIP]' % (block_name,)
            continue
        
        print ' -- %s' % (block_name,)
        
        ws_np_prepare_cmd = ('%s %d %d %d' % 
            (WS_NP_PREPARE_PATH,
             block_depth,
             block_row_id,
             block_col_id))
        
        proc_log_filepath = os.path.join(
            META_DIR, LOG_DIR, LOG_PROCS_DIR, phase_name, '%s_ws_np_prepare.log' % (block_name,))        
            
        ctx.jobs.execute(
            cmd = ws_np_prepare_cmd,
            log_path = proc_log_filepath,
            cmd_env = None)
    
    ctx.jobs.sync_all()
    
    run_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_RUNS_DIR, phase_name, 'ws_np_prepare_summary.log')        
    
    ctx.jobs.report_status(run_log_filepath)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'WS_NP_PREPARE_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        

def ANALYZE_BLOCKS_verify(log_path):
    res = is_run_log_success(log_path)
    return res
    
def ANALYZE_BLOCKS_execute(ctx):
    
    print 'ANALYZE_BLOCKS_execute: start'
    
    phase_name = 'analyze_blocks'
    
    start_time_secs = time.time()
    
    ctx.jobs.init(
        max_jobs = N_ANALYZE_BLOCKS_WORKERS_PER_CPU * N_CPUS)
    
    ctx.jobs.set_proc_verify_func(ANALYZE_BLOCKS_verify)
    
    for block in ctx.blocks_to_process:
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth,
            block_row_id,
            block_col_id)
            
        print ' -- %s' % (block_name,)
            
        analyze_blocks_cmd = ('%s %d %d %d' % 
            (ANALYZE_BLOCKS_BIN_PATH,
             block_depth,
             block_row_id,
             block_col_id))
        
        proc_log_filepath = os.path.join(
            META_DIR, LOG_DIR, LOG_PROCS_DIR, phase_name, '%s_analyze_blocks.log' % (block_name,))        
        
        ctx.jobs.execute(
            cmd = analyze_blocks_cmd,
            log_path = proc_log_filepath,
            cmd_env = None)    
        
    ctx.jobs.sync_all()
    
    run_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_RUNS_DIR, phase_name, 'analyze_blocks_summary.log')        
    
    ctx.jobs.report_status(run_log_filepath)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'ANALYZE_BLOCKS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        
    
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
    
def GENERATE_BLOCKS_verify(log_path):
    res = is_run_log_success(log_path)
    return res
    
def GENERATE_BLOCKS_execute(ctx):
    
    print 'GENERATE_BLOCKS_execute: start'
    
    phase_name = 'generate_blocks'
    
    start_time_secs = time.time()
    
    ctx.jobs.init(
        max_jobs = N_GENERATE_BLOCKS_WORKERS_PER_CPU * N_CPUS)
    
    ctx.jobs.set_proc_verify_func(GENERATE_BLOCKS_verify)
        
    for data_dir in ctx.data_dir_list:
        out_em_dir = os.path.join(data_dir, EM_DIR)
        verify_out_dir(ctx, out_em_dir)
            
    sec_id_start = ctx.sections_to_process[0]
    sec_id_final = ctx.sections_to_process[-1]
    
    while (sec_id_start < sec_id_final):
        
        sec_id_finish = sec_id_start + GENERATE_BLOCKS_SECTION_BATCH
        if sec_id_finish > sec_id_final:
            sec_id_finish = sec_id_final
        
        print ' -- SECTION_IDS     : %d-%d' % (sec_id_start, sec_id_finish)
        
        generate_blocks_cmd = ('%s %d %d' % 
            (GENERATE_BLOCKS_BIN_PATH,
             sec_id_start,
             sec_id_finish))
        
        proc_log_filepath = os.path.join(
            META_DIR, LOG_DIR, LOG_PROCS_DIR, phase_name, 'generate_blocks_sec_ids_%.4d_%.4d.log' % (
                sec_id_start,
                sec_id_finish))
                
        ctx.jobs.execute(
            cmd = generate_blocks_cmd,
            log_path = proc_log_filepath,
            cmd_env = None)
        
        sec_id_start += GENERATE_BLOCKS_SECTION_BATCH
        
    ctx.jobs.sync_all()
    
    run_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_RUNS_DIR, phase_name, 'generate_blocks_summary.log')        
    
    ctx.jobs.report_status(run_log_filepath)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'GENERATE_BLOCKS_execute: finish [total_time = %d secs]' % (elapsed_time_secs,)        

def DEBUG_GENERATE_verify(log_path):
    res = is_run_log_success(log_path)
    return res
    
def DEBUG_GENERATE_exec(ctx):
    
    print 'DEBUG_GENERATE_exec: start'
    
    start_time_secs = time.time()
    
    ctx.jobs.init(
        max_jobs = N_DEBUG_GENERATE_WORKERS_PER_CPU * N_CPUS)
    
    ctx.jobs.set_proc_verify_func(DEBUG_GENERATE_verify)
    
    for block in ctx.blocks_to_process:
        
        block_depth = block[0]
        block_row_id = block[1]
        block_col_id = block[2]
        
        block_name = get_block_name(
            block_depth, 
            block_row_id, 
            block_col_id)
        
        if not meta_is_block_valid(
            block_depth, 
            block_row_id, 
            block_col_id):
            print ' -- %s is not valid [SKIP]' % (block_name,)
            continue
                
        block_debug_path = get_block_debug_path(    
            block_depth, 
            block_row_id, 
            block_col_id)
            
        if not os.path.exists(block_debug_path):
            os.makedirs(block_debug_path)    
        
        print ' -- %s' % (block_name,)
            
        print ' -- DEBUG : %s' % (block_debug_path,)
        
        debug_generate_cmd = ('%s %d %d %d' %
            (DEBUG_GENERATE_PATH,
             block_depth, 
             block_row_id, 
             block_col_id))
        
        proc_log_filepath = os.path.join(
            META_DIR, LOG_DIR, LOG_PROCS_DIR, DEBUG_DIR, '%s_debug_generate.log' % (block_name,))        

        ctx.jobs.execute(
            cmd = debug_generate_cmd,
            log_path = proc_log_filepath,
            cmd_env = None)
    
    ctx.jobs.sync_all()
    
    run_log_filepath = os.path.join(
        META_DIR, LOG_DIR, LOG_RUNS_DIR, DEBUG_DIR, 'debug_summary.log')        
    
    ctx.jobs.report_status(run_log_filepath)
    
    elapsed_time_secs = time.time() - start_time_secs
    
    print 'DEBUG_GENERATE_exec: finish [total_time = %d secs]' % (elapsed_time_secs,)
    

