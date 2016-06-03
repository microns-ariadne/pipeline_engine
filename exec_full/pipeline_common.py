
import os
import sys
import time
import shutil
import subprocess

from pipeline_config import *

###############################################################################
# Pipeline General Context
###############################################################################

class PipelineContext:
    def __init__(self):
        self.jobs = PipelineJobsContext()
        
    def set_input_params(
        self,
        is_force,
        phase,
        sections_to_process,
        blocks_to_process):
        
        self.is_force = is_force
        self.phase = phase
        self.sections_to_process = sections_to_process
        self.blocks_to_process = blocks_to_process
    
    def set_data_dirs(
        self,
        data_dir_list):
        self.data_dir_list = data_dir_list
    
    
###############################################################################
# Pipeline Jobs Context
###############################################################################

class PipelineJobsContext:
    def __init__(self):
        pass
    
    def init(
        self,
        max_jobs):
        
        self.max_jobs = max_jobs
        self.cur_job_id = 0
        self.procs = []
        self.procs_data = []
        
        self.proc_verify_func = None
        self.n_success = 0
        self.n_failed = 0
        self.success_cmds = []
        self.failed_cmds = []
        
    def set_proc_verify_func(
        self,
        func):
        
        self.proc_verify_func = func
    
    def report_status(
        self,
        log_path):
        
        data_str = '\n'
        data_str += '----------------------------------------------------------\n'
        data_str += '-- RUN REPORT \n'
        data_str += '----------------------------------------------------------\n'
        data_str += '-- n_total_cmds   : %d\n' % (self.cur_job_id,)
        data_str += '-- n_success_cmds : %d\n' % (self.n_success,)
        data_str += '-- n_failed_cmds  : %d\n' % (self.n_failed,)
        
        if len(self.failed_cmds) > 0:
            data_str += '----------------------------------------------------------\n'
            data_str += '-- FAILED CMD LIST\n'
            
            for i, failed_cmd in enumerate(self.failed_cmds):
                data_str += '[%d] CMD: %s\n' % (i, failed_cmd)
            
        print data_str
        
        if len(self.success_cmds) > 0:
            data_str += '----------------------------------------------------------\n'
            data_str += '-- SUCCESS CMD LIST\n'
            
            for i, success_cmd in enumerate(self.success_cmds):
                data_str += '[%d] CMD: %s\n' % (i, success_cmd)
        
        data_str += '----------------------------------------------------------\n'
        data_str += '----------------------------------------------------------\n'
        
        assert_out_dir(os.path.dirname(log_path))
        
        print '-- Writing run log: %s' % (log_path,)
        f = open(log_path, 'wb')
        f.write(data_str)
        f.close()
        
        print '\n\n'
        
    def sync(
        self, 
        is_all):
        
        n_procs = len(self.procs)

        print ' -- sync start: %d PROCESSES RUNNING [IS_ALL = %d]' % (n_procs, is_all)

        while (True):

            if not is_all:
                if (len(self.procs) < self.max_jobs):
                    break
            else:
                if (len(self.procs) == 0):
                    break

            time.sleep(0.2)

            for i, proc in enumerate(self.procs):
                if proc.poll() != None:
                    self.procs.remove(proc)

                    entry = self.procs_data[i]
                    self.procs_data.remove(entry)
                    
                    (cmd, job_id, log_path, start_time) = entry
                                        
                    elapsed_time = time.time() - start_time

                    print ' -- jobs_sync: PROCESS %d DONE' % (job_id,)
                    print '    -- CMD: %s' % (cmd,)
                    print '    -- TIME: %d [secs]' % (elapsed_time,)
                    
                    if self.proc_verify_func is not None:
                        print '    -- VERIFY_SUCCESS: start'
                        res = self.proc_verify_func(log_path)
                        print '    -- VERIFY_SUCCESS: done'
                        
                        if res:
                            self.n_success += 1
                            self.success_cmds.append(cmd)
                            print '      -- CMD_SUCCESS'
                            print '        -- LOG: %s' % (log_path,)
                        else:
                            self.n_failed += 1
                            self.failed_cmds.append(cmd)
                            print '      -- CMD_FAILED'
                            print '        -- LOG: %s' % (log_path,)
                    
                    
        print ' -- sync finish: %d PROCESSES RUNNING' % (len(self.procs),)
    
    def sync_all(self):
        self.sync(True)
        
    def sync_max(self):
        self.sync(False)
        
    def exec_process(
        self,
        cmd, 
        log_path, 
        cmd_env = None):

        print 'exec_process: '
        print ' CMD: %s' % (cmd,)
        print ' LOG: %s' % (log_path,)
        
        if cmd_env is None:
            cmd_end = os.environ.copy()
        
        assert_out_dir(os.path.dirname(log_path))
        
        f_log = open(log_path, 'wb', 0)
        
        start_time = time.time()
        
        proc = subprocess.Popen(
            cmd,
            shell = True,
            stdin = None,
            stdout = f_log,
            stderr = f_log,
            env = cmd_env)
            
        self.procs.append(proc)
        
        self.procs_data.append((
            cmd, 
            self.cur_job_id,
            log_path, 
            start_time))
        
        self.cur_job_id += 1
    
    def get_numa_prefix(self):
        cpu_id = self.cur_job_id % N_CPUS
        return 'numactl --cpunodebind=%d' % (cpu_id,)
    
    def execute(
        self,
        cmd,
        log_path,
        cmd_env):
        
        self.sync_max()
        
        numa_cmd = '%s %s' % (self.get_numa_prefix(), cmd)
        
        proc = self.exec_process(
            numa_cmd, 
            log_path, 
            cmd_env)
        
        print 'execute: %d' % (len(self.procs),)
        
        
###############################################################################
# Exec functions
###############################################################################

def exec_cmd(
    cmd,
    cmd_env = None):
    
    print 'EXEC START: %s' % (cmd,)

    start_time_secs = time.time()
    
    proc = subprocess.Popen(
        cmd,
        shell = True,
        stdin = None,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        env = cmd_env)
    
    print '--------------------------------------------------------------------'
    print '-- PROC OUTPUT: start'
    print '--------------------------------------------------------------------'
    out_lines = []
    is_success = False
    while True:
        next_line = proc.stdout.readline()
        if next_line == '' and proc.poll() is not None:
            break
        sys.stdout.write(next_line)
        sys.stdout.flush()
        out_lines.append(next_line)
        if (next_line.find(PROC_SUCCESS_STR) != -1):
            is_success = True
    
        
    print '--------------------------------------------------------------------'
    print '-- PROC OUTPUT: finish'
    print '--------------------------------------------------------------------'
    
    elapsed_time_secs = time.time() - start_time_secs
    
    err_lines = proc.stderr.readlines()
    
    if len(err_lines) > 0:
        print '--------------------------------------------------------------------'
        print '-- PROC ERROR OUTPUT: start'
        print '--------------------------------------------------------------------'
        for err_line in err_lines:
            sys.stdout.write(err_line)
            sys.stdout.flush()
        
        print '--------------------------------------------------------------------'
        print '-- PROC ERROR OUTPUT: finish'
        print '--------------------------------------------------------------------'
        
    print 'EXEC FINISH: Execution time is %s seconds' % (elapsed_time_secs,)
    
    return (is_success, out_lines)

###############################################################################
# Align functions
###############################################################################

def get_align_result_dir(
    sec_id):
    
    data_idx = sec_id % len(DATA_DIR_LIST)
    
    result_dir_path = os.path.join(DATA_DIR_LIST[data_idx], ALIGN_DIR, ALIGN_RESULT_DIR)
    
    return result_dir_path

###############################################################################
# Dir functions
###############################################################################

def clean_dir(dirpath):
    print '-- clean_dir: deleting %s' % (dirpath,)
    shutil.rmtree(dirpath)
    os.makedirs(dirpath)
    
def user_clean_dir(
    ctx, 
    dirpath):
    
    res = is_user_verify(ctx, 'Is clean dir: %s' % (dirpath,))
    
    if not res:
        print '-- clean_dir: skip'
        return
    
    clean_dir(dirpath)
    
def assert_out_dir(out_dirpath):    
    print 'assert_out_dir: start'
    if not os.path.exists(out_dirpath):
        print 'assert_out_dir: creating %s' % (out_dirpath,)
        os.makedirs(out_dirpath)
    
    print 'assert_out_dir: finish'

def verify_out_dir(
    ctx, 
    out_dirpath):
    
    print 'verify_out_dir: start'
    if not os.path.exists(out_dirpath):
        print 'verify_out_dir: creating %s' % (out_dirpath,)
        os.makedirs(out_dirpath)
    else:
        user_clean_dir(
            ctx, 
            out_dirpath)
    
    print 'verify_out_dir: finish'
    
def verify_block_out_dir(out_block_dirpath):
    
    print 'verify_block_out_dir: start'
    if not os.path.exists(out_block_dirpath):
        print 'verify_block_out_dir: creating %s' % (out_block_dirpath,)
        os.makedirs(out_block_dirpath)
    else:
        clean_dir(out_block_dirpath)
    
    print 'verify_block_out_dir: finish'

def get_block_dir_name(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_name = 'block_%.4d_%.4d_%.4d' % (block_depth, block_row_id, block_col_id)
    
    return block_name
        
def get_block_data_dir(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_idx = (block_row_id * BLOCKS_MAX_COL_ID) + block_col_id
    
    data_idx = block_idx % len(DATA_DIR_LIST)
    
    return DATA_DIR_LIST[data_idx]    

def get_block_em_path(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_data_dir = get_block_data_dir(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_dir_name = get_block_dir_name(
        block_depth, 
        block_row_id, 
        block_col_id)    
    
    block_em_path = os.path.join(block_data_dir, EM_DIR, block_dir_name)
    
    return block_em_path
    
def get_block_meta_path(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_data_dir = get_block_data_dir(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_dir_name = get_block_dir_name(
        block_depth, 
        block_row_id, 
        block_col_id)    
    
    block_meta_path = os.path.join(block_data_dir, BLOCKS_META_DIR, block_dir_name)
    
    return block_meta_path

def get_block_probs_path(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_data_dir = get_block_data_dir(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_dir_name = get_block_dir_name(
        block_depth, 
        block_row_id, 
        block_col_id)    
    
    block_probs_path = os.path.join(block_data_dir, PROBS_DIR, block_dir_name)
    
    return block_probs_path

def get_block_probs_processed_path(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_data_dir = get_block_data_dir(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_dir_name = get_block_dir_name(
        block_depth, 
        block_row_id, 
        block_col_id)    
    
    block_probs_path = os.path.join(block_data_dir, PROBS_PROCESSED_DIR, block_dir_name)
    
    return block_probs_path

def get_block_probs_ws_path(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_data_dir = get_block_data_dir(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_dir_name = get_block_dir_name(
        block_depth, 
        block_row_id, 
        block_col_id)    
    
    block_probs_ws_path = os.path.join(block_data_dir, PROBS_WS_DIR, block_dir_name)
    
    return block_probs_ws_path

def get_block_probs_np_path(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_data_dir = get_block_data_dir(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_dir_name = get_block_dir_name(
        block_depth, 
        block_row_id, 
        block_col_id)    
    
    block_probs_np_path = os.path.join(block_data_dir, PROBS_NP_DIR, block_dir_name)
    
    return block_probs_np_path

def get_block_ws_path(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_data_dir = get_block_data_dir(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_dir_name = get_block_dir_name(
        block_depth, 
        block_row_id, 
        block_col_id)    
    
    block_ws_path = os.path.join(block_data_dir, WS_DIR, block_dir_name)
    
    return block_ws_path

def get_block_np_path(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_data_dir = get_block_data_dir(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_dir_name = get_block_dir_name(
        block_depth, 
        block_row_id, 
        block_col_id)    
    
    block_np_path = os.path.join(block_data_dir, NP_DIR, block_dir_name)
    
    return block_np_path

def get_block_merge_path(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_data_dir = get_block_data_dir(
        block_depth, 
        block_row_id, 
        block_col_id)
        
    block_merge_path = os.path.join(block_data_dir, MERGE_DIR)
    
    return block_merge_path

def get_block_out_merge_path(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_merge_path = get_block_merge_path(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_out_merge_path = os.path.join(block_merge_path, 'out_segmentation_%.4d_%.4d_%.4d' % (
        block_depth,
        block_row_id,
        block_col_id))
        
    return block_out_merge_path

def get_merge_meta_path():
    
    return os.path.join(META_DIR, MERGE_DIR)

def get_merge_data_paths():
    
    merge_data_dirs = []
    for data_dir in DATA_DIR_LIST:
        merge_data_dirs.append(os.path.join(data_dir, MERGE_DIR))
    
    return merge_data_dirs

def get_scatter_points_path():
    
    return os.path.join(META_DIR, SCATTER_POINTS_DIR)
    
def get_skeletons_path():
    
    return os.path.join(META_DIR, SKELETONS_DIR)
    
def verify_scatter_points_dirs():
    
    scatter_points_path = get_scatter_points_path()
    
    if not os.path.exists(scatter_points_path):
        os.makedirs(scatter_points_path)
    else:
        clean_dir(scatter_points_path)
        
    full_scatter_points_path = os.path.join(scatter_points_path, 'full')
    
    if not os.path.exists(full_scatter_points_path):
        os.makedirs(full_scatter_points_path)
    else:
        clean_dir(full_scatter_points_path)
    
    boundary_scatter_points_path = os.path.join(scatter_points_path, 'boundary')
    
    if not os.path.exists(boundary_scatter_points_path):
        os.makedirs(boundary_scatter_points_path)
    else:
        clean_dir(boundary_scatter_points_path)
    
def verify_block_skeletons_dirs(skeletons_path):
        
    if not os.path.exists(skeletons_path):
        os.makedirs(skeletons_path)
    else:
        clean_dir(skeletons_path)
        
    SWC_skeletons_path = os.path.join(skeletons_path, 'SWC')
    
    if not os.path.exists(SWC_skeletons_path):
        os.makedirs(SWC_skeletons_path)
    else:
        clean_dir(SWC_skeletons_path)


def get_block_skeletons_path(
    block_depth,
    block_row_id,
    block_col_id):
    
    skeletons_path = get_skeletons_path()
    
    block_name = get_block_name(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_skeletons_path = os.path.join(skeletons_path, block_name)
    
    verify_block_skeletons_dirs(block_skeletons_path)
        
    return block_skeletons_path
    
def get_block_debug_path(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_data_dir = get_block_data_dir(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    block_dir_name = get_block_dir_name(
        block_depth, 
        block_row_id, 
        block_col_id)    
    
    block_debug_path = os.path.join(block_data_dir, DEBUG_DIR, block_dir_name)
    
    return block_debug_path

###############################################################################
# Meta functions
###############################################################################
def meta_update_block_status_file(
    block_depth, 
    block_row_id, 
    block_col_id,
    status):
    
    block_meta_path = get_block_meta_path(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    status_path = os.path.join(
        block_meta_path, 
        META_BLOCK_STATUS_FILE % (
            block_depth,
            block_row_id,
            block_col_id))
    
    print ' -- set block_%.4d_%.4d_%.4d status to %s' % (
        block_depth, 
        block_row_id, 
        block_col_id,
        status)
     
    f = open(status_path, 'wb')
    f.write(status + '\n')
    f.close()

def meta_get_block_status(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_meta_path = get_block_meta_path(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    status_path = os.path.join(
        block_meta_path, 
        META_BLOCK_STATUS_FILE % (
            block_depth,
            block_row_id,
            block_col_id))
    
    if not os.path.exists(status_path):
        return META_BLOCK_STATUS_NOT_VALID
    
    f = open(status_path, 'rb')
    data = f.readlines()
    f.close()
    
    status_str = data[0].strip()
    
    return status_str
    
def meta_is_block_valid(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    status_str = meta_get_block_status(
        block_depth, 
        block_row_id, 
        block_col_id)
        
    return (status_str == META_BLOCK_STATUS_VALID)
        
###############################################################################
# Block functions
###############################################################################    
def get_block_name(
    block_depth, 
    block_row_id, 
    block_col_id):

    return get_block_dir_name(
        block_depth, 
        block_row_id, 
        block_col_id)

def get_block_em_depth_size(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    n_images = BLOCK_N_IMAGES
    if block_depth > BLOCKS_MIN_DEPTH:
        n_images += Z_PAD
    
    if block_depth < BLOCKS_MAX_DEPTH:
        n_images += Z_PAD
    
    return n_images
    
def get_block_cnn_depth_size(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_depth_size = get_block_em_depth_size(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    return block_depth_size - 2

def get_block_em_row_size(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    n_rows = BLOCK_N_ROWS
    
    if block_row_id > BLOCKS_MIN_ROW_ID:
        n_rows += CNN_PATCH_LEG + X_PAD
    
    if block_row_id < BLOCKS_MAX_ROW_ID:
        n_rows += CNN_PATCH_LEG + X_PAD
        
    return n_rows

def get_block_cnn_row_size(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_row_size = get_block_em_row_size(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    return (block_row_size - (2 * CNN_PATCH_LEG)) / BLOCK_SUB_SAMPLE

def get_block_em_col_size(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    n_cols = BLOCK_N_COLS
    
    if block_col_id > BLOCKS_MIN_COL_ID:
        n_cols += CNN_PATCH_LEG + Y_PAD
    
    if block_col_id < BLOCKS_MAX_COL_ID:
        n_cols += CNN_PATCH_LEG + Y_PAD
    
    return n_cols
    
def get_block_cnn_col_size(
    block_depth, 
    block_row_id, 
    block_col_id):
    
    block_col_size = get_block_em_col_size(
        block_depth, 
        block_row_id, 
        block_col_id)
    
    return (block_col_size - (2 * CNN_PATCH_LEG)) / BLOCK_SUB_SAMPLE

def get_Z_pad_cut_indices(
    block_depth,
    block_size,
    z_pad_right_overlap):
    
    start_idx = 0
    finish_idx = block_size
    
    z_pad_left = Z_PAD - 1
    z_pad_right = Z_PAD - 1
    
    assert(z_pad_right > 0)
    z_pad_right -= z_pad_right_overlap
            
    if block_depth > BLOCKS_MIN_DEPTH:
        start_idx += z_pad_left
    
    if block_depth < BLOCKS_MAX_DEPTH:
        finish_idx -= z_pad_right
    
    return (start_idx, finish_idx)

###############################################################################
# Block padding functions
###############################################################################

def get_X_pad_cut_indices(
    block_row_id,
    block_rows):
    
    start_idx = 0
    finish_idx = block_rows
    
    if block_row_id > BLOCKS_MIN_ROW_ID:
        start_idx += (X_PAD / BLOCK_SUB_SAMPLE)
    
    if block_row_id < BLOCKS_MAX_ROW_ID:
        finish_idx -= (X_PAD / BLOCK_SUB_SAMPLE)
    
    return (start_idx, finish_idx)

def get_Y_pad_cut_indices(
    block_col_id,
    block_cols):
    
    start_idx = 0
    finish_idx = block_cols
    
    if block_col_id > BLOCKS_MIN_COL_ID:
        start_idx += (Y_PAD / BLOCK_SUB_SAMPLE)
    
    if block_col_id < BLOCKS_MAX_COL_ID:
        finish_idx -= (Y_PAD / BLOCK_SUB_SAMPLE)
    
    return (start_idx, finish_idx)

###############################################################################
# Exec verify functions
###############################################################################    

def is_run_log_success(log_path):
    f = open(log_path, 'rb')
    lines = f.readlines()
    f.close()
    
    last_lines = lines[-3:]
    
    for line in last_lines:
        if (line.find(PROC_SUCCESS_STR) != -1):
            return True
    
    return False

###############################################################################
# Interactive functions
###############################################################################    

def user_verify(
    ctx, 
    is_abort,
    msg):
    
    q_msg = '%s ? [YES/NO] ' % (msg,)
    
    if ctx.is_force:
        print '%s' % (q_msg,) 
        print ' => FORCE YES'
        return
    
    yes_list = ['YES', 'yes', 'Y', 'y']
    no_list = ['NO', 'no', 'N', 'n']
    
    while (True):
        in_res = raw_input(q_msg)
        
        if in_res in yes_list:
            print ' => CONFIRMED'
            return True
        
        if in_res in no_list:
            print ' => DECLINED'
            if is_abort:
                raise Exception('ABORTING: declined user_verify = %s' % (msg,))
            else:
                return False

def is_user_verify(
    ctx, 
    msg):
    
    res = user_verify(
        ctx,
        False,
        msg)
    
    return res

def assert_user_verify(
    ctx, 
    msg):
    
    user_verify(
        ctx,
        True,
        msg)
    
